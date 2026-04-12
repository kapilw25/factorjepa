# FactorJEPA — Research Update (3 slides)

---

## Slide 1: What Failed + Why (Ch10 Pretrain)

### 💀 Ch10 Result: CATASTROPHIC FORGETTING

| Metric | Frozen Baseline | Ch10 Adapted | Delta |
|---|---|---|---|
| Prec@K | 36.1% ±0.6 | **14.3%** ±0.3 | **-21.8pp** ❌ |
| nDCG@K | 0.950 ±0.001 | 0.906 ±0.001 | -0.045 ❌ |
| mAP@K | 0.278 ±0.006 | 0.080 ±0.002 | -0.198 ❌ |

### 🔍 Root Causes (Gold Standard Audit: 12 fixes)

| What Was Wrong | What We Fixed |
|---|---|
| ❌ lambda=0.001 (1000x too weak) | ✅ lambda=100 (EWC range) |
| ❌ Predictor LR 10x encoder | ✅ 1x (Meta uses same LR) |
| ❌ Masked-only loss (50% signal missing) | ✅ Dense loss on ALL tokens |
| ❌ Single-layer supervision | ✅ 4-layer deep supervision (6656-dim) |
| ❌ V-JEPA 2.0 (1B, 1408-dim) | ✅ V-JEPA 2.1 (2B, 1664-dim) |
| ❌ Cosine LR decay | ✅ Constant LR (warmup then flat) |
| ❌ grad_clip=1.0 | ✅ 10.0 (V-JEPA 1/2 default) |

---

## Slide 2: New Approach — ExPLoRA + Surgery (NOT pretrain)

### 🎯 Strategy Pivot: Skip Ch10 → Go directly to Ch11

| | 🐢 Old (Ch10 Pretrain) | ⚡ New (ExPLoRA) | 🔪 New (Surgery) |
|---|---|---|---|
| **What** | Brute-force retrain ALL layers | LoRA + unfreeze 2 blocks | Factor-decomposed progressive unfreezing |
| **Model** | V-JEPA 2.0 (1B) | **V-JEPA 2.1 (2B)** | **V-JEPA 2.1 (2B)** |
| **Data** | Raw Indian clips | Raw Indian clips | **D_L (layout) + D_A (agents) + D_I (interactions)** |
| **Params** | All 1B trainable | **~5% trainable** | **25%→50%→75% progressive** |
| **Anti-forget** | lambda=0.001 (failed) | Freezing + LoRA | **3-stage curriculum + replay** |
| **Loss** | Masked-only L1 | **Dense L1 + deep supervision** | **Dense L1 + deep supervision** |
| **Time** | 6h (failed) | **~1h** | **~35 min** |
| **Novelty** | None (standard) | Low (ExPLoRA exists) | **HIGH (factor decomposition is NEW)** |

### 🏗️ Surgery Pipeline (THE paper novelty)

```
SAM 3.1 → "auto_rickshaw, pedestrian" per clip (from tags.json)
  ↓
D_L: blur agents (layout-only) → Stage 1: teach layers 0-12
D_A: suppress BG (agent-only)  → Stage 2: teach layers 0-24 + 10% D_L replay
D_I: interaction tubes          → Stage 3: teach layers 0-36 + 10% D_A + 5% D_L
  ↓
student_encoder.pt → re-embed → Prec@K comparison
```

---

## Slide 3: Execution Plan — 1K Val Clips, ~70 min GPU

### 📋 Ready to Run NOW

| Step | Command | Time | Status |
|---|---|---|---|
| 1b | `./scripts/train_explora.sh --POC` | ~1h | ✅ READY |
| 2 | `./scripts/train_surgery.sh --POC` | ~35 min | ✅ READY |

### 🏆 Decision Gate

| ExPLoRA | Surgery | Action |
|---|---|---|
| ✅ Improves | 🏆 Surgery > ExPLoRA | **Best Paper: factor surgery wins** |
| ✅ Improves | = ExPLoRA | Publish ExPLoRA result |
| ❌ No change | 🏆 Surgery improves | **Strongest novelty: standard fails, surgery succeeds** |
| ❌ No change | ❌ No change | Fallback: SIGReg (LeJEPA), leakage-free (VLA-JEPA) |

### 🆕 Why This Will Succeed (vs Ch10 failure)

| Ch10 (failed) | ExPLoRA + Surgery (new) |
|---|---|
| 1B model | **2B model** (+23.5 mIoU spatial) |
| 50% training signal | **100% training signal** (dense loss) |
| 1-layer supervision | **4-layer deep supervision** |
| All layers trainable = overwrite everything | **Progressive freeze = protect what works** |
| Raw clips = kitchen sink | **Factor-decomposed = teach one concept at a time** |
| No SAM, no tags | **SAM 3.1 per-clip prompts from VLM tags** |

### 📊 24 JEPA variants surveyed → FactorJEPA STILL NOVEL
> "No one decomposes INPUT videos into semantic factors for JEPA training."
