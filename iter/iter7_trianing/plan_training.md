# Training Plan: Ch10 (Continual Pretraining) + Ch11 (Surgery Fine-Tuning)

> Ref: `Literature/proposal/FactorJEPA/FactorJEPA.md` Sections 10-11

---

## V-JEPA Training: What's Actually Used

PPO/DPO/GRPO are RLHF methods for text-generating LLMs. They are **fundamentally inapplicable** to V-JEPA. V-JEPA is a deterministic encoder (video → embedding), not a generative model. There's no reward signal, no preference pairs, no policy to optimize.

### V-JEPA 2.0 vs 2.1 Training Components

| Component | V-JEPA 2.0 | V-JEPA 2.1 |
|-----------|-----------|-----------|
| **Loss** | L1 latent prediction (masked tokens only) | Dense Predictive Loss (ALL tokens, L1) |
| **Optimizer** | AdamW | AdamW |
| **LR Schedule** | Warmup-constant-cooldown (NOT cosine) | Same |
| **EMA** | Fixed momentum (no ramp-up) | Same |
| **Architecture** | Student-teacher with predictor | Same + deep self-supervision at intermediate layers |

Sources: [V-JEPA 2 (arXiv:2506.09985)](https://arxiv.org/abs/2506.09985), [V-JEPA 2.1 (arXiv:2603.14482)](https://arxiv.org/abs/2603.14482)

---

## Self-Supervised Video Encoder Training Algorithms

| Algorithm | Loss Type | Used By | Negatives? |
|-----------|-----------|---------|------------|
| **JEPA latent prediction (L1)** | Regression in latent space | V-JEPA 2/2.1 | No |
| DINO + iBOT | Cross-entropy (CLS + patch) | DINOv2 | No (EMA teacher) |
| MSE pixel reconstruction | Pixel regression | VideoMAE, MAE | No |
| BYOL | MSE normalized projection | BYOL | No (EMA) |
| InfoNCE / NT-Xent | Contrastive | SimCLR, MoCo | Yes |

---

## Continual Pretraining Approaches (Ch10)

Proposal (Sec 10.3) specifies: same JEPA loss on Indian clips, student-teacher EMA, optional drift control.

Standard approaches in literature:

| # | Approach | How it works | Relevance |
|---|----------|-------------|-----------|
| 1 | **Same SSL loss on new data** | Resume pretraining with JEPA loss on Indian clips | Most direct. V-JEPA 2 itself does stage-wise training (pretrain → post-train). **Our primary approach.** |
| 2 | **EWC (Elastic Weight Consolidation)** | Penalty on important weights from prior training | Prevents catastrophic forgetting. Our drift control (λ·‖θ-θ₀‖²) is equivalent to L2-anchored EWC. |
| 3 | **Knowledge distillation** | Frozen original model as teacher, adapted model matches teacher outputs + learns from new data | Confirmed for CLIP/DINOv2 continual learning. Could supplement JEPA loss. |
| 4 | **LoRA / adapters** | Freeze backbone, train low-rank adapter modules | Reduces trainable params. C-LoRA confirmed for continual vision learning. |
| 5 | **Frozen encoder + new predictor** | Freeze encoder, train only predictor on new data | V-JEPA 2's own action-conditioned post-training uses this. Cheapest option. |

---

## Surgery Fine-Tuning Approaches (Ch11)

Proposal (Sec 11.5) specifies: progressive prefix unfreezing with factor datasets (Layout → Agent → Interaction).

| Stage | Layers Unfrozen | Input | Factor |
|-------|----------------|-------|--------|
| 1 | 0 to n₁ (~25% of L) | 100% D_L (layout-only) | Roads, buildings, wires |
| 2 | 0 to n₂ (~50% of L) | 90% D_A + 10% D_L replay | Vehicles, people, animals |
| 3 | 0 to n₃ (~75% of L) | 85% D_I + 10% D_A + 5% D_L | Agent-agent interactions |

Factor datasets (D_L, D_A, D_I) created via SAM3 segmentation → tracklet mining → agent/layout separation.

---

## Python Packages with JEPA Training Code

| Package | JEPA Support | Status |
|---------|-------------|--------|
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | **YES** — full training configs in `configs/train/vitg16/` | Active, official |
| [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | **YES** — V-JEPA 1 training (`app/vjepa/train.py`) | Active |
| [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa) | **YES** — lightweight JEPA examples (CIFAR-10, Moving MNIST) | Active (2026) |
| LightlySSL | No JEPA (has BYOL, DINO, SimCLR, MoCo, MAE) | Active |
| solo-learn | No JEPA | Active |
| VISSL | No JEPA | Archived (2024) |

**For Ch10/Ch11**: Use Meta's official `facebookresearch/vjepa2` training code. Configs exist at `configs/train/vitg16/` (2.0) and `configs/train_2_1/vitG16/` (2.1 ablation).

---

## Model Strategy

| Phase | Model | Dim | Purpose |
|-------|-------|-----|---------|
| 10K POC | V-JEPA 2.0 ViT-g (1B) | 1408 | Validate pipeline works |
| 115K Primary | V-JEPA 2.0 ViT-g (1B) | 1408 | Main results, proposal-aligned |
| 115K Ablation | V-JEPA 2.1 ViT-G (2B) | 1664 | Scaling analysis for paper |

---

## Ch10 Training Recipe (from proposal Sec 10.3-10.5)

```
1. Load V-JEPA 2 ViT-g checkpoint (student + teacher via deepcopy)
2. Stream Indian clips (uniform by video_id, stratified by v3 taxonomy)
3. Per step:
   a. Decode T frames, resize to 384px
   b. Generate context + target views (video-consistent augmentations)
   c. Sample spatiotemporal masks (15-30%, 2-6 blocks)
   d. Student forward on context view (visible tokens)
   e. Teacher forward on target view (masked tokens, no grad)
   f. Predictor maps student features → teacher space
   g. L1 loss on masked token predictions + drift control
   h. AdamW step on student + predictor
   i. EMA update teacher: θ̄ ← τ·θ̄ + (1-τ)·θ
4. Checkpoint every 2K-5K steps
5. Select best checkpoint by Cycle@K (hard mode) on validation subset
```

---

## Code Organization

### Why flat `src/m*.py` (not subdirectories)

Researched V-JEPA 2, DINOv2, VideoMAE, timm, MMAction2, SAM 2 repo structures. Two patterns dominate:

- **V-JEPA 2 / DINOv2**: `app/` (train) + `evals/` (eval) subdirectories — works when train and eval are **independent programs** with separate entry points, data loaders, dependencies
- **VideoMAE / timm**: Flat scripts with naming convention (`run_mae_pretraining.py`, `run_class_finetuning.py`)

**Our pipeline is sequential, not independent.** Modules share `config.py`, `gpu_batch.py`, encoder registry, WebDataset streaming, `bootstrap.py`, `wandb_utils.py`. m05 outputs feed m06 which feeds m08. Splitting into subdirectories would break the `m04 → m05 → m06 → m07 → m08` dependency chain that the numbering encodes.

### Decision: Flat modules + config-driven training (V-JEPA 2 style)

```
src/
├── m00-m08b                 # Existing eval pipeline (KEEP FLAT)
├── m09_pretrain.py          # Ch10 continual pretraining
├── m10_sam_segment.py       # Ch11 SAM3 + tracklets
├── m11_factor_datasets.py   # Ch11 D_L/D_A/D_I creation
├── m12_surgery.py           # Ch11 progressive unfreezing
└── utils/                   # Shared (KEEP)

configs/                     # NEW — V-JEPA 2 style
├── eval/
│   └── vitg16_poc.yaml
├── pretrain/
│   └── vitg16_indian.yaml
└── surgery/
    ├── stage1_layout.yaml
    ├── stage2_agent.yaml
    └── stage3_interaction.yaml

scripts/
├── run_evaluate.sh          # Ch9 (EXISTS)
├── run_pretrain.sh          # Ch10 (TODO)
└── run_surgery.sh           # Ch11 (TODO)
```

YAML configs carry hyperparameters (LR, EMA momentum, mask ratio, batch size, prefix boundary). Python scripts carry logic. This matches V-JEPA 2 / DINOv2 / MMAction2 conventions without breaking our module numbering.

---

## Implementation Path

### Training infrastructure from `facebookresearch/vjepa2`

| Their file | Purpose | Our adaptation |
|-----------|---------|---------------|
| `app/main.py` | Single-GPU training entry | `src/m09_pretrain.py` (same role) |
| `app/main_distributed.py` | Multi-GPU DDP | `src/m09_pretrain.py --distributed` (flag, not separate file) |
| `configs/train/vitg16/pretrain-256px-16f.yaml` | Base pretrain config | `configs/pretrain/vitg16_indian.yaml` |
| `configs/train/vitg16/cooldown-256px-64f.yaml` | Cooldown config | `configs/pretrain/vitg16_cooldown.yaml` |
| `src/models/vision_transformer.py` | ViT-g/ViT-G architecture | Import directly from vjepa2 package |
| `src/masks/` | Spatiotemporal mask sampling | Import directly |
| `src/datasets/` | Video data loading | Replace with our `_create_stream()` + `--local-data` pattern |

### Our modifications for Indian continual pretraining

1. **Data loader**: Replace their `src/datasets/` with HF WebDataset streaming (existing `_create_stream()` from m05)
2. **Drift control**: Add regularizer λ·‖θ-θ₀‖² to JEPA loss (Sec 10.4 of proposal)
3. **Stratified sampling**: Uniform by `video_id`, balanced by v3 taxonomy tags
4. **Validation loop**: Run Cycle@K (hard mode) on held-out video_ids every 2K-5K steps
5. **Checkpoint selection**: Best Cycle@K, not lowest loss (label-free selection)
6. **Logging**: Integrate with `utils/wandb_utils.py`
7. **Config-driven**: All hyperparameters in YAML, script reads config via `--config configs/pretrain/vitg16_indian.yaml`
