# Code Development Plan — Week 1 Experiments

> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**

---

## Completed (this session)

| Task | File | What |
|---|---|---|
| Config restructure | `configs/model/` + `configs/train/` | 6 YAML files: vjepa2_0, vjepa2_1, base_optimization, ch10_pretrain, explora, ch11_surgery |
| Config merging | `src/utils/config.py` | `load_merged_config()`, `get_model_config()`, `add_model_config_arg()`, `add_train_config_arg()` |
| Zero hardcoding | `src/utils/config.py` + `configs/pipeline.yaml` | VJEPA_MODEL_ID, EMBEDDING_DIM, ENCODER_REGISTRY, VLM_MODELS, CLIP_MIN/MAX all from YAML |
| V-JEPA 2.1 import | `src/utils/vjepa2_imports.py` | `get_vit_gigantic_xformers()` + `get_vit_by_arch(arch)` dispatcher |
| m05 V-JEPA 2.1 | `src/m05_vjepa_embed.py` | `--model-config` flag, native 2.1 checkpoint loading, 3 paths: adapted / native-frozen / HF-frozen |
| Temporal projection | `src/m06c_temporal_projection.py` | **IMPLEMENTED** — PCA sweep k=[5..200], verify_or_skip, resume, vectorized cosine, atomic saves |
| Checkpoint utils | `src/utils/checkpoint.py` | Generic atomic save/load (embedding, array, JSON) |
| CLAUDE.md rules | `src/CLAUDE.md` | Rules 29 (GPU infra checklist), 30 (--model-config + --train-config), 31 (get_vit_by_arch) |
| 3-check gate | `.claude/hooks/post-edit-lint.sh` | py_compile + AST + ruff F+E9, venv ruff preferred, m06c in cpu_only set |
| pyyaml fix | `requirements.txt` | Added PyYAML (was missing, broke `uv pip sync`) |

---

## TODO #1: m09_pretrain.py — ExPLoRA + V-JEPA 2.1 Support

**File:** `src/m09_pretrain.py` (~940 lines)
**Priority:** CRITICAL — blocks Step 1b (ExPLoRA baseline)

### Changes (7 edits, ~80 lines)

#### 1. Argparse: replace `--config` with `--model-config` + `--train-config` (line 1327-1328)

```python
# Keep --config for backward compat with train_pretrain.sh
parser.add_argument("--config", type=str, default=None, help="Legacy YAML config path")
add_model_config_arg(parser)    # --model-config configs/model/vjepa2_1.yaml
add_train_config_arg(parser)    # --train-config configs/train/explora.yaml
parser.add_argument("--explora", action="store_true", help="Enable ExPLoRA mode")
```

#### 2. Config loading: use `load_merged_config()` (line 1361-1362)

```python
if args.config:
    cfg = load_config(args.config)  # Legacy path
elif args.train_config:
    cfg = load_merged_config(args.model_config, args.train_config)
else:
    print("FATAL: Specify --config or --train-config")
    sys.exit(1)
cfg = merge_config_with_args(cfg, args)
if args.explora:
    cfg.setdefault("explora", {})["enabled"] = True
```

#### 3. build_model(): use `get_vit_by_arch()` (line 305)

```python
# OLD: vit_giant_xformers = get_vit_giant_xformers()
from utils.vjepa2_imports import get_vit_by_arch
arch = cfg["model"]["arch"]
vit_constructor = get_vit_by_arch(arch)
```

#### 4. build_model(): checkpoint from YAML (line 325-337)

```python
# OLD: hardcoded URL "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt"
ckpt_url = cfg["model"]["checkpoint_url"]
ckpt_local = project_root / cfg["model"]["checkpoint_path"]
```

#### 5. build_model(): ExPLoRA injection (INSERT after line ~430)

```python
explora_cfg = cfg.get("explora")
if explora_cfg and explora_cfg.get("enabled"):
    from peft import get_peft_model, LoraConfig

    for param in student.parameters():
        param.requires_grad = False
    for i in range(explora_cfg["unfreeze_blocks"]):
        for param in student.blocks[i].parameters():
            param.requires_grad = True
    if explora_cfg["unfreeze_norm_layers"]:
        for name, param in student.named_parameters():
            if "norm" in name or "ln" in name:
                param.requires_grad = True

    lora_config = LoraConfig(
        r=explora_cfg["lora_rank"],
        lora_alpha=explora_cfg["lora_alpha"],
        target_modules=explora_cfg["lora_target_modules"],
        lora_dropout=explora_cfg["lora_dropout"],
        bias="none",
    )
    student = get_peft_model(student, lora_config)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"  ExPLoRA: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    init_params = {name: p.clone().detach().cpu()
                   for name, p in student.named_parameters()
                   if not p.requires_grad and "lora" not in name}
```

#### 6. update_teacher_ema(): name-safe EMA (line 508-514)

```python
@torch.no_grad()
def update_teacher_ema(student, teacher, momentum):
    student_dict = dict(student.named_parameters())
    for name, param_t in teacher.named_parameters():
        if name in student_dict:
            param_t.mul_(momentum).add_(student_dict[name].data, alpha=1.0 - momentum)
```

#### 7. export_student_for_eval(): merge LoRA before save (line 687-696)

```python
def export_student_for_eval(student, path, explora_enabled=False):
    if explora_enabled:
        student = student.merge_and_unload()
    torch.save({"student_state_dict": student.state_dict(), ...}, path)
```

### PEFT compatibility risk

vjepa2's ViT uses custom xformers attention. PEFT targets `nn.Linear` modules by name.
Need to verify `student.blocks[i].attn.qkv` and `student.blocks[i].attn.proj` are `nn.Linear`.

**Fallback if PEFT fails:** Manual LoRALinear wrapper (~30 lines, no external dep).

### Dependency

Add to `requirements_gpu.txt`: `peft>=0.13.0`

---

## TODO #2: Shell Scripts for ExPLoRA + Surgery

**Pattern:** Separate scripts (matches codebase style — one script per pipeline).

### Create `scripts/train_explora.sh` (~200 lines)

Copy-modify from `train_pretrain.sh`. Key differences:
- Calls m09 with `--model-config configs/model/vjepa2_1.yaml --train-config configs/train/explora.yaml --explora`
- No lambda ablation (ExPLoRA uses no drift control)
- Output to `outputs/{mode}/m09_explora/`
- Re-embed with `--encoder vjepa_2_1_explora`

```bash
#!/bin/bash
# ExPLoRA: LoRA + unfreeze 1-2 blocks + JEPA pretraining on V-JEPA 2.1
source scripts/lib/common.sh

run_step "explora" "m09 ExPLoRA" "$T_TRAIN" "$LOGDIR/m09_explora.log" \
    src/m09_pretrain.py \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/train/explora.yaml --explora \
        $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG $VAL_FLAG --no-wandb

run_step "embed" "m05 re-embed ExPLoRA" "$T_EMBED" "$LOGDIR/m05_explora.log" \
    src/m05_vjepa_embed.py --model-config configs/model/vjepa2_1.yaml \
        --model "${OUT_DIR}/m09_explora/student_encoder.pt" \
        --encoder vjepa_2_1_explora $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb

run_step "eval" "m06 metrics" "$T_EVAL" "$LOGDIR/m06_explora.log" \
    src/m06_faiss_metrics.py --encoder vjepa_2_1_explora $MODE_FLAG $SUBSET_FLAG --no-wandb

finalize
```

### Fill `scripts/train_surgery.sh` (~250 lines)

Currently a 136-line stub (comments only, no shell code). Fill with:

```bash
#!/bin/bash
# Ch11: Factor Surgery — SAM3 segmentation + progressive prefix unfreezing on V-JEPA 2.1
source scripts/lib/common.sh

# Step 0 (CPU): SAM3 → factor datasets
run_step "sam3" "m10 SAM3 segmentation" "$T_SAM3" "$LOGDIR/m10_sam3.log" \
    src/m10_sam_segment.py $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb

run_step "factors" "m11 factor datasets" "$T_FACTORS" "$LOGDIR/m11_factors.log" \
    src/m11_factor_datasets.py $MODE_FLAG $SUBSET_FLAG --no-wandb

# Step 1 (GPU): 3-stage progressive unfreezing
run_step "surgery" "m09 surgery" "$T_SURGERY" "$LOGDIR/m09_surgery.log" \
    src/m09_pretrain.py \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/train/ch11_surgery.yaml \
        $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG $VAL_FLAG --no-wandb

# Step 2: Re-embed + eval
run_step "embed" "m05 re-embed surgical" "$T_EMBED" "$LOGDIR/m05_surgery.log" \
    src/m05_vjepa_embed.py --model-config configs/model/vjepa2_1.yaml \
        --model "${OUT_DIR}/m09_surgery/student_encoder.pt" \
        --encoder vjepa_2_1_surgical $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb

run_step "eval" "m06 metrics" "$T_EVAL" "$LOGDIR/m06_surgery.log" \
    src/m06_faiss_metrics.py --encoder vjepa_2_1_surgical $MODE_FLAG $SUBSET_FLAG --no-wandb

finalize
```

**Note:** m10 and m11 Python scripts don't exist yet. The shell script provides orchestration skeleton.

### Why separate scripts (not --mode flag in train_pretrain.sh)

| Pattern | Pros | Cons |
|---|---|---|
| **Separate (chosen)** | Focused, readable. Matches codebase style. Future divergence doesn't create conflicts. | ~80 lines of boilerplate duplicated from train_pretrain.sh |
| Extend train_pretrain.sh | Single file, less duplication. | Grows to 500+ lines, conditional branches everywhere, less discoverable |

Existing codebase: `run_embed.sh`, `run_eval.sh`, `train_pretrain.sh` are all separate. `train_surgery.sh` stub already exists.

---

## TODO #3: Low Priority

| Task | Priority | Notes |
|---|---|---|
| m05c V-JEPA 2.1 | LOW | True Overlap@K — not needed for Week 1 |
| m09 Ch11 surgery mode | AFTER Step 2 plan | Needs m10/m11 Python scripts first |

---

## V-JEPA 2.1 Checkpoint Download (GPU instance)

```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt -P checkpoints/
ls -lh checkpoints/vjepa2_1_vitG_384.pt  # Expect ~8 GB
```

---

## Q3: MAX GPU Utilization on RTX 3090 (24 GB)

3-layer system:

1. **Static estimate** (`compute_batch_sizes()`): Linear scale from 40GB baseline. RTX 3090 → BS=9.
2. **Profiler** (`get_optimal_batch_size()`): Reads profile_data.json, finds max BS at <=75% VRAM.
3. **Runtime** (`AdaptiveBatchSizer`): Monitors `torch.cuda.mem_get_info()` every batch. Grows at <65%, shrinks at >85%, halves on OOM.

**RTX 3090 (24GB) → Yes, 80% utilization.** AdaptiveBatchSizer auto-adapts. Wall clock ~4x slower than 96GB but GPU stays busy. V-JEPA 2.1 (2B): student+teacher = ~8GB bf16, leaves ~16GB for activations → BS ~4-8 with checkpointing.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| PEFT incompatible with vjepa2 ViT | ExPLoRA blocked | Manual LoRALinear wrapper (~30 lines) |
| V-JEPA 2.1 (2B) OOMs on 96GB | Can't train | activation_checkpointing + smaller BS. Fallback: 2.1 ViT-g 1B (`vjepa2_1_vitg_384.pt`) |
| 2.1 checkpoint key mismatch | Load fails | m09 handles prefix stripping + strict=False |
| ExPLoRA shows 0 improvement | No baseline to beat | Best outcome for Ch11: "standard fails, surgery succeeds" |

---

## Execution Order (remaining)

| # | Task | Est. |
|---|---|---|
| 1 | m09: argparse + config loading changes | 20 min |
| 2 | m09: build_model() — get_vit_by_arch + checkpoint from YAML | 10 min |
| 3 | m09: ExPLoRA injection block in build_model() | 15 min |
| 4 | m09: update_teacher_ema + export_student_for_eval | 10 min |
| 5 | m09: 3-check gate (py_compile + ruff + AST) | 5 min |
| 6 | scripts/train_explora.sh | 15 min |
| 7 | scripts/train_surgery.sh (fill stub) | 15 min |
| **Total** | | **~90 min** |
