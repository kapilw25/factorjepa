# Code Development Plan — m09 Surgery Mode (Progressive Prefix Unfreezing)

> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**
> **This is the LAST blocker. Everything else is built and ready.**

---

## What's Blocked

`train_surgery.sh` calls `m09_pretrain.py --train-config configs/train/ch11_surgery.yaml` but m09 has NO code to:
1. Read the `surgery.stages` config
2. Load factor datasets (D_L/D_A/D_I) from `--factor-dir`
3. Iterate over stages with expanding trainable prefix
4. Sample training data according to per-stage mode mixture
5. Rebuild optimizer per stage (new trainable params)
6. Per-stage warmup

---

## Design: `train_surgery()` function in m09

**NOT a new file.** Surgery reuses 95% of the existing training loop — same JEPA loss, same masking, same EMA, same checkpointing. The only differences are:

1. **Data source**: Factor-patched clips from m11 (D_L/D_A/D_I .npy files) instead of raw clips
2. **Trainable layers**: Expand at each stage boundary
3. **Data mixture**: Changes at each stage boundary
4. **Optimizer**: Rebuild at each stage boundary (new params enter)

### Architecture

```python
def train_surgery(cfg: dict, args):
    """3-stage progressive prefix unfreezing with factor datasets (Ch11)."""

    # 1. Build model (same as train())
    models = build_model(cfg, device)
    student, teacher, predictor, init_params = ...

    # 2. Load factor dataset paths
    factor_dir = Path(args.factor_dir)
    dl_dir = factor_dir / "D_L"
    da_dir = factor_dir / "D_A"
    di_dir = factor_dir / "D_I"
    manifest = json.load(open(factor_dir / "factor_manifest.json"))

    # 3. Build factor clip index: {clip_key: {D_L: path, D_A: path, D_I: [paths]}}
    factor_index = build_factor_index(manifest, dl_dir, da_dir, di_dir)

    # 4. Read surgery stages from config
    stages = cfg["surgery"]["stages"]
    depth = cfg["model"]["depth"]  # 48 for ViT-G

    # 5. Per-stage loop
    for stage_cfg in stages:
        stage_name = stage_cfg["name"]
        unfreeze_frac = stage_cfg["unfreeze_below"]
        mode_mixture = stage_cfg["mode_mixture"]
        warmup_steps = stage_cfg["warmup_steps"]
        epoch_pct = stage_cfg["max_epochs_pct"]

        # 5a. Set trainable prefix
        n_trainable = int(depth * unfreeze_frac)
        set_trainable_prefix(student, n_trainable)

        # 5b. Rebuild optimizer (only trainable params)
        optimizer = build_optimizer(student, predictor, cfg["optimization"])

        # 5c. Build stage data sampler (mode_mixture weights)
        sampler = FactorSampler(factor_index, mode_mixture)

        # 5d. Compute stage steps
        stage_steps = int(total_steps * epoch_pct)

        # 5e. Per-stage warmup scheduler
        scheduler = build_warmup_scheduler(optimizer, warmup_steps, stage_steps)

        # 5f. Training loop (reuse existing step logic)
        for step in range(stage_steps):
            factor_type, clip_data = sampler.sample()
            # ... same forward/backward as train(), but clip_data from .npy
```

---

## New Functions Needed in m09

### 1. `set_trainable_prefix(student, n_layers)` (~15 lines)

```python
def set_trainable_prefix(student, n_layers: int):
    """Freeze all blocks, then unfreeze blocks [0, n_layers).

    PyTorch best practice: set requires_grad THEN rebuild optimizer.
    Ref: https://discuss.pytorch.org/t/154297
    """
    # Freeze everything
    for param in student.parameters():
        param.requires_grad = False

    # Unfreeze first n_layers blocks
    for i in range(n_layers):
        for param in student.blocks[i].parameters():
            param.requires_grad = True

    # Always unfreeze norm layers (ExPLoRA + surgery both need this)
    for name, param in student.named_parameters():
        if "norm" in name or "ln" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"  Trainable prefix: {n_layers}/{len(student.blocks)} blocks "
          f"({trainable:,}/{total:,} params = {100*trainable/total:.1f}%)")
```

### 2. `build_factor_index(manifest, dl_dir, da_dir, di_dir)` (~25 lines)

```python
def build_factor_index(manifest: dict, dl_dir: Path, da_dir: Path, di_dir: Path) -> dict:
    """Build clip_key → factor file paths mapping from m11 manifest."""
    index = {}
    for clip_key, info in manifest.items():
        safe_key = clip_key.replace("/", "__")
        entry = {}
        if info["has_D_L"]:
            entry["D_L"] = dl_dir / f"{safe_key}.npy"
        if info["has_D_A"]:
            entry["D_A"] = da_dir / f"{safe_key}.npy"
        if info.get("has_D_I") and info.get("n_interaction_tubes", 0) > 0:
            entry["D_I"] = sorted(di_dir.glob(f"{safe_key}_tube*.npy"))
        index[clip_key] = entry
    return index
```

### 3. `FactorSampler` class (~40 lines)

```python
class FactorSampler:
    """Sample factor clips according to stage mode mixture weights."""

    def __init__(self, factor_index: dict, mode_mixture: dict):
        self.mode_mixture = mode_mixture  # {"L": 0.9, "A": 0.1, "I": 0.0}
        self.factor_map = {"L": "D_L", "A": "D_A", "I": "D_I"}

        # Build per-factor clip lists
        self.clips_by_factor = {}
        for factor_key in ["D_L", "D_A", "D_I"]:
            self.clips_by_factor[factor_key] = [
                (key, entry[factor_key])
                for key, entry in factor_index.items()
                if factor_key in entry
            ]

        # Precompute weights for np.random.choice
        self.factors = []
        self.weights = []
        for mode_key, weight in mode_mixture.items():
            if weight > 0:
                factor_key = self.factor_map[mode_key]
                if self.clips_by_factor[factor_key]:
                    self.factors.append(factor_key)
                    self.weights.append(weight)
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]

    def sample(self) -> tuple:
        """Returns (factor_type, clip_data_path)."""
        factor = np.random.choice(self.factors, p=self.weights)
        clips = self.clips_by_factor[factor]
        idx = np.random.randint(len(clips))
        clip_key, path = clips[idx]
        if isinstance(path, list):  # D_I has multiple tubes
            path = path[np.random.randint(len(path))]
        return factor, clip_key, path
```

### 4. `load_factor_clip(path)` (~10 lines)

```python
def load_factor_clip(path: Path, num_frames: int, crop_size: int) -> torch.Tensor:
    """Load a factor-patched clip from .npy and prepare for training."""
    frames = np.load(path)  # (T, H, W, C) uint8
    # Resize if needed (D_I tubes may have different sizes)
    if frames.shape[1] != crop_size or frames.shape[2] != crop_size:
        from PIL import Image as PILImage
        resized = []
        for t in range(frames.shape[0]):
            img = PILImage.fromarray(frames[t])
            img = img.resize((crop_size, crop_size), PILImage.BILINEAR)
            resized.append(np.array(img))
        frames = np.stack(resized)
    # Select num_frames uniformly
    if frames.shape[0] > num_frames:
        indices = np.linspace(0, frames.shape[0] - 1, num_frames, dtype=int)
        frames = frames[indices]
    elif frames.shape[0] < num_frames:
        pad = np.repeat(frames[-1:], num_frames - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    # Convert to (T, C, H, W) float tensor, normalize
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return tensor
```

### 5. `train_surgery()` main function (~120 lines)

The key insight: the inner training step is IDENTICAL to `train()`. The difference is:
- Data comes from `.npy` files (not WebDataset streaming)
- Trainable prefix expands at each stage boundary
- Optimizer rebuilt per stage
- No drift control (surgery uses freezing instead)

```python
def train_surgery(cfg: dict, args):
    """3-stage progressive prefix unfreezing with factor datasets."""
    check_gpu()
    device = torch.device("cuda")

    # Seeds + model setup (same as train())
    ...
    models = build_model(cfg, device)
    student = models["student"]
    teacher = models["teacher"]
    predictor = models["predictor"]

    # Load factor datasets
    factor_dir = Path(args.factor_dir)
    manifest = json.load(open(factor_dir / "factor_manifest.json"))
    factor_index = build_factor_index(manifest, ...)

    # Surgery stages from config
    surgery_cfg = cfg["surgery"]
    stages = surgery_cfg["stages"]
    depth = cfg["model"]["depth"]
    total_epochs = cfg["optimization"]["max_epochs"][args.mode_key]
    total_clips = len(manifest)
    batch_size = cfg["optimization"]["batch_size"]
    steps_per_epoch = total_clips // batch_size
    total_steps = steps_per_epoch * total_epochs

    for stage_idx, stage_cfg in enumerate(stages):
        stage_name = stage_cfg["name"]
        n_trainable = int(depth * stage_cfg["unfreeze_below"])
        stage_steps = int(total_steps * stage_cfg["max_epochs_pct"])
        warmup_steps = stage_cfg["warmup_steps"]

        print(f"\n{'='*60}")
        print(f"SURGERY STAGE {stage_idx + 1}/{len(stages)}: {stage_name}")
        print(f"  Trainable: layers 0-{n_trainable} of {depth}")
        print(f"  Steps: {stage_steps}")
        print(f"  Mixture: {stage_cfg['mode_mixture']}")
        print(f"{'='*60}")

        # Set trainable prefix
        set_trainable_prefix(student, n_trainable)

        # Rebuild optimizer (only trainable params)
        optimizer = build_optimizer(student, predictor, cfg["optimization"])

        # Build factor sampler for this stage
        sampler = FactorSampler(factor_index, stage_cfg["mode_mixture"])

        # Per-stage warmup scheduler
        scheduler = build_warmup_scheduler(optimizer, warmup_steps, stage_steps, cfg)

        # Training loop (same forward/backward as train())
        for step in range(stage_steps):
            # Sample factor clip
            factor_type, clip_key, clip_path = sampler.sample()

            # Load batch of factor clips
            batch_clips = []
            for _ in range(batch_size):
                ft, ck, cp = sampler.sample()
                clip_tensor = load_factor_clip(cp, cfg["data"]["num_frames"], cfg["model"]["crop_size"])
                batch_clips.append(clip_tensor)
            batch = torch.stack(batch_clips).to(device)
            batch = batch.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) for ViT

            # Generate masks (same as train())
            all_masks_enc, all_masks_pred = [], []
            for mg in mask_generators:
                m_enc, m_pred = mg(batch_size)
                all_masks_enc.append(m_enc.to(device))
                all_masks_pred.append(m_pred.to(device))

            # Forward + loss (IDENTICAL to train() — dense loss + deep supervision)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=True):
                with torch.no_grad():
                    h = teacher(batch)
                    # Per-chunk LayerNorm (deep supervision)
                    ...

                pred_features, pred_context = [], []
                for i, (m_enc, m_pred) in enumerate(...):
                    z = student(batch, masks=[m_enc])
                    outputs = predictor(z, [m_enc], [m_pred], mask_index=i)
                    ...

                jepa_loss, loss_masked, loss_context = compute_jepa_loss(...)

            # Backward + step (same as train())
            scaler.scale(jepa_loss).backward()
            ...
            update_teacher_ema(student, teacher, ema_momentum)

            # Logging (per step)
            ...

        # End of stage: save checkpoint
        save_training_checkpoint(...)

    # Export final student
    export_student_for_eval(student, output_dir / "student_encoder.pt")
```

---

## CLI Changes in m09

### New argparse flags

```python
parser.add_argument("--surgery", action="store_true",
                    help="Enable surgery mode (progressive prefix unfreezing + factor datasets)")
parser.add_argument("--factor-dir", type=str, default=None,
                    help="Factor dataset directory from m11 (contains D_L/, D_A/, D_I/, factor_manifest.json)")
```

### Updated main() dispatch

```python
if args.surgery:
    if not args.factor_dir:
        print("FATAL: --surgery requires --factor-dir")
        sys.exit(1)
    train_surgery(cfg, args)
elif args.explora:
    ...
else:
    train(cfg, args)
```

### Update train_surgery.sh

Remove the FATAL guard, add `--surgery --factor-dir`:

```bash
run_step "2-surgery" "m09 factor surgery" "$T_SURGERY" "$LOGDIR/m09_surgery.log" \
    src/m09_pretrain.py \
        --model-config "$MODEL_CONFIG" \
        --train-config "$TRAIN_CONFIG" \
        --surgery --factor-dir "${FACTOR_DIR}" \
        --output-dir "$SURGERY_DIR" \
        $BATCH_FLAG $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG $VAL_FLAG --no-wandb
```

---

## Other Blockers Assessment

| Item | Status | Blocker? |
|---|---|---|
| V-JEPA 2.1 checkpoint download | In setup_env_uv.sh | NO |
| SAM 3.1 install | In runbook.md | NO |
| peft install | In runbook.md | NO |
| Dense loss | DONE | NO |
| Deep supervision | DONE | NO |
| return_hierarchical | DONE | NO |
| 2.1 imports from app/vjepa_2_1/ | DONE | NO |
| m10 SAM 3.1 API | DONE | NO |
| m11 factor datasets | DONE | NO |
| Mask feathering | DONE | NO |
| Quality filters | DONE | NO |
| train_explora.sh safety | DONE | NO |
| **m09 surgery mode** | **NOT DONE** | **YES — ONLY blocker** |

**No other blockers.** Once `train_surgery()` is implemented in m09, the full pipeline runs end-to-end:

```
train_surgery.sh --POC
  → m10 (SAM 3.1 on 1K clips, ~5 min)
  → m11 (D_L + D_A + D_I, ~3 min)
  → m09 --surgery (3-stage unfreezing, ~15 min)
  → m05 (re-embed, ~12 min)
  → m06 (eval Prec@K)
  → compare: frozen 2.1 vs ExPLoRA vs surgery
```

---

## TODO List

| # | Task | Est. |
|---|---|---|
| 1 | `set_trainable_prefix()` function | 5 min |
| 2 | `build_factor_index()` function | 5 min |
| 3 | `FactorSampler` class | 10 min |
| 4 | `load_factor_clip()` function | 5 min |
| 5 | `train_surgery()` main function (reuses train() logic) | 30 min |
| 6 | CLI flags: `--surgery`, `--factor-dir` | 5 min |
| 7 | Update `main()` dispatch | 2 min |
| 8 | Remove FATAL guard from `train_surgery.sh`, add `--surgery --factor-dir` | 5 min |
| 9 | 3-check gate (py_compile + ruff + bash -n) | 5 min |
| **Total** | | **~70 min** |

---

## Verification

### Mac (syntax)

```bash
source venv_walkindia/bin/activate
python3 -m py_compile src/m09_pretrain.py
python3 -m ruff check --select F,E9 src/m09_pretrain.py
bash -n scripts/train_surgery.sh
```

### GPU (functional)

```bash
# 1. Run m10 + m11 to generate factor datasets
python -u src/m10_sam_segment.py --SANITY --local-data data/val_1k_local --no-wandb
python -u src/m11_factor_datasets.py --SANITY --local-data data/val_1k_local --no-wandb

# 2. SANITY: surgery on 20 clips (verify no crash)
python -u src/m09_pretrain.py \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/ch11_surgery.yaml \
    --surgery --factor-dir outputs/sanity/factors \
    --SANITY --no-wandb

# 3. POC: full surgery pipeline
./scripts/train_surgery.sh --POC
```

Sources:
- [PyTorch progressive unfreezing](https://discuss.pytorch.org/t/how-to-freeze-all-and-progressively-unfreeze-layers-of-a-model-for-transfert-learning/154297)
- [ExPLoRA (ICML 2025)](https://arxiv.org/abs/2406.10973)
- [ExPLoRA code](https://samar-khanna.github.io/ExPLoRA/)
