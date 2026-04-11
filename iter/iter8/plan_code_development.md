# Plan: Step 1a (Temporal Projection) + Step 1b (ExPLoRA) + V-JEPA 2.1 + Config Restructure

## Context

> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**

Week 1 has 3 parallel workstreams:

- **Step 1a:** Temporal interference projection (30 min CPU, diagnostic) -- **IMPLEMENT FIRST**
- **Step 1b:** ExPLoRA baseline adaptation (3h GPU, bar to beat)

---

## Part -1: Config Restructure (before any code)

### Current structure (flat, not scalable)

```
configs/
├── pipeline.yaml
└── pretrain/
    └── vitg16_indian.yaml       # mixes model + training + everything
```

### New structure (model separate from technique)

```
configs/
├── pipeline.yaml                    # Shared infra: clip limits, streaming, eval, data processing
├── model/
│   ├── vjepa2_0.yaml                # V-JEPA 2.0 ViT-g (1B, 1408-dim)
│   └── vjepa2_1.yaml                # V-JEPA 2.1 ViT-G (2B, 1664-dim)
└── train/
    ├── base_optimization.yaml       # Shared: LR, EMA, warmup, grad_clip, mixed precision
    ├── ch10_pretrain.yaml           # Ch10: JEPA loss + drift control + lambda sweep
    ├── ch11_surgery.yaml            # Ch11: factor datasets + 3-stage progressive unfreezing
    ├── explora.yaml                 # ExPLoRA: LoRA + unfreeze 1-2 blocks
    └── lora.yaml                    # Plain LoRA (fallback)
```

### How configs merge

Each training script takes `--model-config` + `--train-config`:

```bash
python -u src/m09_pretrain.py \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/explora.yaml \
    --FULL ...
```

Python merges: `pipeline.yaml` (base) + `model/*.yaml` + `train/*.yaml` -> single config dict. No duplication.

### What moves from `config.py` to YAML

| Value | From `config.py` | To YAML file |
|---|---|---|
| `VJEPA_MODEL_ID` | line 106 | `configs/model/vjepa2_0.yaml` -> `model.model_id` |
| `VJEPA_EMBEDDING_DIM = 1408` | line 108 | `configs/model/vjepa2_0.yaml` -> `model.embed_dim` |
| `REENCODE_CRF = 28` | line 48 | `configs/pipeline.yaml` -> `data.reencode_crf` |
| `CLIP_MIN_DURATION = 4.0` | line 102 | `configs/pipeline.yaml` -> `data.clip_min_duration` |
| `CLIP_MAX_DURATION = 10.0` | line 103 | `configs/pipeline.yaml` -> `data.clip_max_duration` |

**Keep in Python:** `ENCODER_REGISTRY` (code logic), path constants (derived), `HF_DATASET_REPO` (infra).

### Files to create/modify for restructure

| File | Action |
|---|---|
| `configs/model/vjepa2_0.yaml` | **NEW** -- extract from `vitg16_indian.yaml` |
| `configs/model/vjepa2_1.yaml` | **NEW** -- 2B model params + checkpoint URL |
| `configs/train/base_optimization.yaml` | **NEW** -- shared optimization params |
| `configs/train/ch10_pretrain.yaml` | **NEW** -- extract from `vitg16_indian.yaml` |
| `configs/train/explora.yaml` | **NEW** -- LoRA + unfreeze config |
| `configs/train/ch11_surgery.yaml` | **NEW** -- factor schedule + progressive unfreezing |
| `src/utils/config.py` | Remove hardcoded model params, add `load_merged_config()` |
| `src/m09_pretrain.py` | Use `--model-config` + `--train-config` instead of `--config` |
| `configs/pretrain/vitg16_indian.yaml` | **KEEP as legacy** (backward compat for existing scripts) |

---
- **Step 2:** Ch11 factor surgery POC (3h GPU, THE experiment -- not in this plan)

This plan covers **Step 1a + Step 1b + the shared V-JEPA 2.1 prerequisite.**

---

## Part 0: V-JEPA 2.1 Integration (shared prerequisite)

### Checkpoint availability (confirmed)

V-JEPA 2.1 checkpoints are public:

- **ViT-G/16 (2B, 1664-dim):** `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt`
- **ViT-g/16 (1B, 1408-dim, 2.1 recipe):** `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt`

Architecture: `vit_gigantic_xformers` -- embed_dim=1664, depth=48, num_heads=26
(defined in `deps/vjepa2/src/models/vision_transformer.py:464`)

### V-JEPA 2.1 recipe differences from 2.0

| Aspect | 2.0 (current) | 2.1 (target) |
|---|---|---|
| Loss | L1 on masked tokens only | **Dense loss: L1 on ALL tokens** (`predict_all: true`) |
| Supervision | Final layer only | **Deep supervision at intermediate layers** (`weight_distance_loss: true`) |
| Predictor depth | 12 | **24** (doubled) |
| Architecture (2B) | ViT-g: 1408-dim, 40 blocks | **ViT-G: 1664-dim, 48 blocks** |
| Resolution | 384x384 | 384x384 (same) |
| Frames | 16 train / 64 eval | 16 train / 64 eval (same) |

### Files to modify

| File | Change | Lines (est.) |
|---|---|---|
| `src/utils/config.py` | Add VJEPA_2_1 constants, update ENCODER_REGISTRY | ~15 |
| `src/utils/vjepa2_imports.py` | Add `get_vit_gigantic_xformers()` export | ~10 |
| `src/m05_vjepa_embed.py` | Support 2.1 model loading (2B checkpoint) | ~20 |
| `src/m09_pretrain.py` | Support 2.1 model for training (ExPLoRA + Surgery) | ~25 |
| `configs/pretrain/vitG16_indian_2_1.yaml` | New config for 2.1 training | ~100 (new file) |

### Detailed changes

#### `src/utils/config.py`

```python
# Line 106-108: Add 2.1 model ID + dim
VJEPA_MODEL_ID = "facebook/vjepa2-vitg-fpc64-384"              # 2.0 (HF, for frozen eval)
VJEPA_2_1_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt"
VJEPA_2_1_CHECKPOINT_PATH = "checkpoints/vjepa2_1_vitG_384.pt"  # Local cache
VJEPA_2_1_EMBEDDING_DIM = 1664

# ENCODER_REGISTRY: add vjepa_2_1 entry
"vjepa_2_1": {"model_id": None, "dim": 1664, "type": "video", "suffix": "_vjepa_2_1"},
```

#### `src/utils/vjepa2_imports.py`

Add `get_vit_gigantic_xformers()` alongside existing `get_vit_giant_xformers()`.
The function imports from `src.models.vision_transformer.vit_gigantic_xformers` (line 464 of vjepa2's vision_transformer.py).

#### `src/m05_vjepa_embed.py`

Add `--model-version` flag (default `"2.0"`):

- `"2.0"`: Current HF AutoModel path (unchanged)
- `"2.1"`: Native vjepa2 `vit_gigantic_xformers()` + load from `VJEPA_2_1_CHECKPOINT_PATH`

Model loading for 2.1 follows the same pattern as adapted model loading (lines 575-598) -- instantiate native ViT, load state_dict from checkpoint, strip prefixes.

**Important:** Output embeddings will be `(N, 1664)` not `(N, 1408)`. All downstream code (m06, m07, m08) reads dim dynamically from array shape -- no changes needed.

#### `configs/pretrain/vitG16_indian_2_1.yaml` (NEW)

Copy `vitg16_indian.yaml`, modify:

- `model.arch: vit_gigantic_xformers` (not `vit_giant_xformers`)
- `model.embed_dim: 1664`
- `model.depth: 48`
- `model.pred_depth: 24` (2.1 uses deeper predictor)
- `loss.predict_all: true` (dense loss -- ALL tokens, not just masked)
- `loss.weight_distance_loss: true` (deep supervision at intermediate layers)
- `checkpoint_url`: point to 2.1 ViT-G checkpoint

---

## Part 1: Temporal Interference Projection (Step 1a) -- NEW SCRIPT

### What it does

1. Load `embeddings.npy` (vjepa normal) and `embeddings_vjepa_shuffled.npy` (shuffled)
2. Compute PCA on `(normal - shuffled)` difference vectors
3. Project normal embeddings orthogonal to top-k PCA components
4. Save projected embeddings
5. Run m06 to compute Prec@K on projected embeddings

### New file: `src/m06c_temporal_projection.py`

**CLI:**

```bash
python -u src/m06c_temporal_projection.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06c.log
python -u src/m06c_temporal_projection.py --SANITY 2>&1 | tee logs/m06c_sanity.log
```

**Architecture (~150 lines):**

```python
"""Temporal interference projection: PCA on (normal - shuffled), project out, re-run Prec@K. CPU-only.
    python -u src/m06c_temporal_projection.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06c.log
"""

def compute_temporal_subspace(emb_normal, emb_shuffled, n_components=50):
    """PCA on (normal - shuffled) difference vectors. Returns projection matrix."""
    diffs = emb_normal - emb_shuffled                  # (N, D)
    diffs_centered = diffs - diffs.mean(axis=0)        # Center
    # Truncated SVD (faster than full PCA for top-k)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(diffs_centered)
    components = svd.components_                        # (k, D)
    explained = svd.explained_variance_ratio_
    return components, explained

def project_out(embeddings, components):
    """Project embeddings orthogonal to subspace defined by components."""
    # V = components.T  -> (D, k)
    # P_perp = I - V @ V.T  (orthogonal projector)
    # projected = embeddings @ P_perp = embeddings - embeddings @ V @ V.T
    V = components.T                                    # (D, k)
    projected = embeddings - (embeddings @ V) @ V.T     # (N, D)
    return projected.astype(np.float32)

def main():
    # 1. Parse args (--SANITY/--FULL/--POC, --subset, --n-components)
    # 2. Load embeddings
    emb_normal = np.load(output_dir / "embeddings.npy")
    emb_shuffled = np.load(output_dir / "embeddings_vjepa_shuffled.npy")
    # 3. Verify alignment
    paths_n = np.load(output_dir / "embeddings.paths.npy", allow_pickle=True)
    paths_s = np.load(output_dir / "embeddings_vjepa_shuffled.paths.npy", allow_pickle=True)
    assert np.array_equal(paths_n, paths_s), "Clip key mismatch!"
    # 4. Sweep n_components in [5, 10, 25, 50, 100, 200]
    for k in [5, 10, 25, 50, 100, 200]:
        components, explained = compute_temporal_subspace(emb_normal, emb_shuffled, k)
        projected = project_out(emb_normal, components)
        suffix = f"_temporal_proj_k{k}"
        np.save(output_dir / f"embeddings{suffix}.npy", projected)
        shutil.copy(output_dir / "embeddings.paths.npy",
                    output_dir / f"embeddings{suffix}.paths.npy")
    # 5. Run m06 on each projected encoder (subprocess)
    for k in [5, 10, 25, 50, 100, 200]:
        encoder_name = f"temporal_proj_k{k}"
        subprocess.run([sys.executable, "-u", "src/m06_faiss_metrics.py",
                        "--encoder", encoder_name, mode_flag, ...])
    # 6. Print comparison table: original Prec@K vs projected Prec@K for each k
    # 7. Save summary to m06c_projection_results.json
```

### Key design decisions

- **Sweep n_components** `[5, 10, 25, 50, 100, 200]` -- don't guess, measure
- **Uses `project_out()` (orthogonal projection)** -- not PCA dimensionality reduction. Output stays at full dim (1408 or 1664)
- **Reuses m06 via subprocess** -- no duplicate metric code
- **Dynamic encoder suffix** via `get_encoder_info()` fallback (auto-generates `_temporal_proj_k{k}`)
- **CPU-only** -- no GPU needed (numpy + sklearn TruncatedSVD)
- **Reuse**: `get_output_dir()`, `add_mode_args()`, `make_pbar()` from existing utils

### Inputs (already exist at `outputs/poc/`)

```
outputs/poc/embeddings.npy                      (10000, 1408) float32  <- vjepa normal
outputs/poc/embeddings_vjepa_shuffled.npy        (10000, 1408) float32  <- shuffled
outputs/poc/embeddings.paths.npy                 (10000,) object       <- clip keys (same order)
outputs/poc/embeddings_vjepa_shuffled.paths.npy  (10000,) object       <- same keys
outputs/poc/tags.json                            list[dict]             <- for Prec@K
```

### Outputs

```
outputs/poc/embeddings_temporal_proj_k5.npy      (10000, 1408) <- projected, k=5
outputs/poc/embeddings_temporal_proj_k10.npy     (10000, 1408)
...
outputs/poc/embeddings_temporal_proj_k200.npy    (10000, 1408)
outputs/poc/m06_metrics_temporal_proj_k{k}.json  <- Prec@K results per k
outputs/poc/m06c_projection_results.json         <- summary table
```

---

## Part 2: ExPLoRA Adaptation (Step 1b)

### What it does

ExPLoRA = **freeze all ViT blocks EXCEPT blocks 0-1** + **LoRA (rank 8-16) on frozen blocks** + **continue JEPA self-supervised pretraining** on Indian clips.

Ref: [ExPLoRA (ICML 2025)](https://arxiv.org/abs/2406.10973) -- +8% on DINOv2 domain shift with <10% params.

### Implementation approach: Extend m09_pretrain.py

**Do NOT create a new script.** m09 already has the full JEPA training loop, producer thread, checkpointing, wandb, etc. Add ExPLoRA as a config-driven mode.

### Files to modify

| File | Change | Lines (est.) |
|---|---|---|
| `src/m09_pretrain.py` | Add LoRA injection + block freeze logic in `build_model()` | ~40 |
| `configs/pretrain/vitG16_indian_2_1.yaml` | Add ExPLoRA config section | ~15 |
| `requirements_gpu.txt` | Add `peft>=0.13.0` | 1 |
| `setup_env_uv.sh` | Add peft to GPU deps | 1 |

### Detailed changes to `src/m09_pretrain.py`

#### A. Add LoRA injection after student loading (inside `build_model()`, after line ~375)

```python
# After student weights are loaded, BEFORE optimizer build
explora_cfg = cfg.get("explora")  # None if not using ExPLoRA
if explora_cfg and explora_cfg.get("enabled"):
    from peft import get_peft_model, LoraConfig

    # 1. Freeze all blocks
    for param in student.parameters():
        param.requires_grad = False

    # 2. Unfreeze first N blocks (ExPLoRA recipe: 1-2 blocks)
    n_unfreeze = explora_cfg["unfreeze_blocks"]  # e.g., 2
    for i in range(n_unfreeze):
        for param in student.blocks[i].parameters():
            param.requires_grad = True

    # 3. Unfreeze all norm layers (ExPLoRA requirement)
    for name, param in student.named_parameters():
        if "norm" in name or "ln" in name:
            param.requires_grad = True

    # 4. Add LoRA to frozen attention layers
    lora_config = LoraConfig(
        r=explora_cfg["lora_rank"],           # 8 or 16
        lora_alpha=explora_cfg["lora_alpha"],  # 2 * rank
        target_modules=explora_cfg["lora_target_modules"],  # ["qkv", "proj"]
        lora_dropout=explora_cfg.get("lora_dropout", 0.0),
        bias="none",
    )
    student = get_peft_model(student, lora_config)

    # 5. Log trainable params
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"  ExPLoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")
```

**CRITICAL:** `peft.get_peft_model()` wraps the model. Need to verify:

1. Forward signature still works with vjepa2's mask-based forward
2. `student.parameters()` still returns all params (peft wraps them)
3. `state_dict()` keys compatible with checkpoint save/load

**Alternative if PEFT doesn't work with vjepa2's custom ViT:** Manual LoRA injection -- add `LoRALinear` wrapper to `qkv` and `proj` Linear layers directly. This is ~30 more lines but no external dependency risk.

#### B. Update drift control to skip LoRA params

In `compute_drift_loss()` (line 492-500), LoRA params won't be in `init_params` dict (they didn't exist at init time), so they're automatically excluded. **No change needed.**

#### C. Update EMA teacher

Teacher is a deep copy made BEFORE LoRA injection. Two options:

1. **Don't inject LoRA into teacher** -- teacher stays as original V-JEPA. EMA update skips LoRA params.
2. **Inject LoRA into teacher too** -- EMA updates LoRA params like any other.

**Recommendation:** Option 1 (no LoRA in teacher). Teacher provides stable targets.

**Implementation:** Modify `update_teacher_ema()` to skip params that don't exist in teacher:

```python
@torch.no_grad()
def update_teacher_ema(student, teacher, momentum):
    student_params = dict(student.named_parameters())
    for name, param_t in teacher.named_parameters():
        if name in student_params:
            param_t.mul_(momentum).add_(student_params[name].data, alpha=1.0 - momentum)
```

### New config section in `configs/pretrain/vitG16_indian_2_1.yaml`

```yaml
explora:
  enabled: false                    # Toggle (false = standard training, true = ExPLoRA)
  unfreeze_blocks: 2                # Unfreeze first N blocks (ExPLoRA default: 1-2)
  lora_rank: 16                     # LoRA rank (8-32)
  lora_alpha: 32                    # Scaling (typically 2x rank)
  lora_target_modules:              # Which linear layers get LoRA
    - "qkv"                         # Fused Q/K/V attention
    - "proj"                        # Attention output projection
  lora_dropout: 0.0                 # LoRA dropout (0 for SSL)
```

### ExPLoRA training command

```bash
# Step 1b: ExPLoRA on V-JEPA 2.1
python -u src/m09_pretrain.py \
    --config configs/pretrain/vitG16_indian_2_1.yaml \
    --FULL --subset data/subset_10k.json --local-data data/subset_10k_local \
    --lambda-reg 0 --max-epochs 5 --no-wandb \
    --explora \
    2>&1 | tee logs/m09_explora_2_1.log
```

The `--explora` flag sets `cfg["explora"]["enabled"] = True` at runtime.

### ExPLoRA output

```
outputs/poc/m09_explora_2_1/
├── student_encoder.pt          # Adapted encoder (with LoRA weights merged)
├── training_summary.json       # Loss curves, hyperparams
└── loss_log.jsonl              # Per-step metrics
```

After training, re-embed + evaluate:

```bash
python -u src/m05_vjepa_embed.py --encoder vjepa_2_1_explora --model-version 2.1 \
    --adapted-checkpoint outputs/poc/m09_explora_2_1/student_encoder.pt \
    --FULL --subset data/subset_10k.json --local-data data/subset_10k_local
python -u src/m06_faiss_metrics.py --encoder vjepa_2_1_explora --FULL --subset data/subset_10k.json
```

---

## Execution Order

| # | File | What | Where | Test |
|---|---|---|---|---|
| 0a | Download checkpoint | `wget vjepa2_1_vitG_384.pt` -> `checkpoints/` | GPU | verify file size |
| 0b | `requirements_gpu.txt` | Add `peft>=0.13.0` | Mac | -- |
| 1 | `src/utils/config.py` | Add VJEPA_2_1 constants + ENCODER_REGISTRY entries | Mac | py_compile |
| 2 | `src/utils/vjepa2_imports.py` | Add `get_vit_gigantic_xformers()` | Mac | py_compile |
| 3 | `configs/pretrain/vitG16_indian_2_1.yaml` | New config for 2.1 + ExPLoRA section | Mac | YAML parse |
| 4 | `src/m06c_temporal_projection.py` | **NEW** -- temporal projection script | Mac | py_compile + --help |
| 5 | `src/m09_pretrain.py` | Add ExPLoRA LoRA + block freeze logic | Mac | py_compile + --help |
| 6 | `src/m05_vjepa_embed.py` | Add `--model-version 2.1` support | Mac | py_compile + --help |

**Order rationale:** Config + imports first (dependency), then scripts that use them. m06c is independent of m09/m05 changes.

---

## Verification Plan

### Tier 1: Syntax (M1 Mac)

```bash
source venv_walkindia/bin/activate
for f in src/utils/config.py src/utils/vjepa2_imports.py \
         src/m06c_temporal_projection.py src/m09_pretrain.py src/m05_vjepa_embed.py; do
    python3 -m py_compile "$f" && echo "OK: $f" || echo "FAIL: $f"
done
python3 -c "import yaml; yaml.safe_load(open('configs/pretrain/vitG16_indian_2_1.yaml'))"
```

### Tier 2: --help (M1 Mac)

```bash
python3 src/m06c_temporal_projection.py --help   # verify --n-components, --FULL, --SANITY
python3 src/m09_pretrain.py --help               # verify --explora flag appears
python3 src/m05_vjepa_embed.py --help            # verify --model-version flag appears
```

### Tier 3: Step 1a functional test (M1 Mac, uses existing 2.0 embeddings)

```bash
# Step 1a can run on EXISTING V-JEPA 2.0 embeddings as a smoke test
# (will re-run on 2.1 embeddings later on GPU)
python3 src/m06c_temporal_projection.py --SANITY 2>&1 | tee logs/m06c_sanity.log
# Verify: outputs/sanity/embeddings_temporal_proj_k5.npy exists
# Verify: outputs/sanity/m06c_projection_results.json exists
```

### Tier 4: GPU integration (RTX PRO 6000)

```bash
# 0. Download 2.1 checkpoint
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt -P checkpoints/

# 1. Step 1a on 2.0 POC embeddings (already exist, 30 min CPU)
python -u src/m06c_temporal_projection.py --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m06c.log

# 2. Generate 2.1 frozen embeddings (needed for Step 1b comparison)
python -u src/m05_vjepa_embed.py --model-version 2.1 --FULL --subset data/subset_10k.json \
    --local-data data/subset_10k_local 2>&1 | tee logs/m05_vjepa_2_1.log

# 3. Step 1b: ExPLoRA training (3h GPU)
python -u src/m09_pretrain.py --config configs/pretrain/vitG16_indian_2_1.yaml \
    --FULL --subset data/subset_10k.json --local-data data/subset_10k_local \
    --lambda-reg 0 --max-epochs 5 --explora --no-wandb \
    2>&1 | tee logs/m09_explora_2_1.log

# 4. Re-embed + evaluate ExPLoRA adapted model
python -u src/m05_vjepa_embed.py --encoder vjepa_2_1_explora --model-version 2.1 \
    --adapted-checkpoint outputs/poc/m09_explora_2_1/student_encoder.pt \
    --FULL --subset data/subset_10k.json --local-data data/subset_10k_local
python -u src/m06_faiss_metrics.py --encoder vjepa_2_1_explora --FULL --subset data/subset_10k.json
```

---

## Dependencies

| Package | Version | Purpose | Install |
|---|---|---|---|
| `peft` | >=0.13.0 | LoRA injection for ExPLoRA | `requirements_gpu.txt` |
| `sklearn` | (already installed) | TruncatedSVD for temporal projection | -- |

No other new dependencies. V-JEPA 2.1 uses the same vjepa2 submodule code.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| PEFT incompatible with vjepa2's custom ViT forward | ExPLoRA blocked | Manual LoRA injection (LoRALinear wrapper, ~30 lines, no external dep) |
| V-JEPA 2.1 (2B) OOMs on 96GB GPU | Can't train | Use activation_checkpointing + smaller batch size. Fallback: 2.1 ViT-g (1B, 1408-dim, `vjepa2_1_vitg_384.pt`) |
| 2.1 checkpoint key format differs from 2.0 | Model loading fails | m09 already handles prefix stripping + strict=False. Add key mapping if needed. |
| Temporal projection shows no effect | Step 1a uninformative | Expected for ~50% chance. Doesn't affect Step 1b or Step 2. |
| ExPLoRA shows 0 improvement | No baseline to beat | Actually the BEST outcome for Ch11 -- "standard adaptation fails, surgery succeeds" |

---

## VRAM Estimate for V-JEPA 2.1 (2B)

| Component | Size |
|---|---|
| Student (2B params, bf16) | ~4 GB |
| Teacher (2B params, bf16, frozen) | ~4 GB |
| Predictor (24-depth, 384-dim) | ~0.3 GB |
| Optimizer states (AdamW, 2x student trainable) | ~0.5 GB (ExPLoRA: <10% params trainable) |
| Activations (with checkpointing) | ~10-20 GB |
| **Total** | **~20-30 GB** |

Should fit on 96GB RTX PRO 6000 with room to spare. Batch size ~16-32 (profile first).
