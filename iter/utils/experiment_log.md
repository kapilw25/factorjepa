# Experiment Log — FactorJEPA Continual Pretraining

> Append-only. Record ALL hyperparameters — we lost 16f/64f info once, never again.

---

## Run 2026-04-05: λ=0.001, 1 epoch, 115K — CATASTROPHIC FORGETTING

**Links**:
- Code: [`b5c04b4`](https://github.com/kapilw25/factorjepa/commit/b5c04b4)
- Training wandb: *(lost — logs deleted before extracting URL)*
- Eval wandb: [runs/7tw2t6sq](https://wandb.ai/sjsu_llm/walkindia-200k/runs/7tw2t6sq)

### Hyperparameters

**pretrain/vitg16_indian.yaml**: arch=vit_giant_xformers (1408d, depth=40, 22 heads), pred=12 layers/384d/12 heads, RoPE=true, activation_ckpt=true | frames=16, crop=384, patch=16, tubelet=2, seed=42 | lr=1e-5, pred_lr=10x, warmup=500 (cap 10%), min_lr=1e-7, wd=0.04, betas=[0.9,0.999], grad_clip=1.0, ema=0.99925, loss_exp=1.0, bf16=true | **λ=0.001**, L2 drift | mask: 8×[0.15] + 2×[0.7] spatial, [1.0] temporal | augment: resize=[0.3,1.0], ratio=[0.75,1.35], hflip=0.5, ImageNet norm=YES | ckpt: 10/epoch, keep=2 | epochs: sanity=1, poc=1, full=1

**pipeline.yaml**: train_bs=112, eval_frames=64, adapted_bs=44, frozen_bs=176, faiss_k=6, ckpt_every=500, decode_workers=16, prefetch=8

### Training Output

clips=114,576, bs=112, steps=1023 (=114576/112), epochs=1.0 | jepa_loss=0.476, drift_loss=0.00047, total=0.477 | best_val=1.648, final_lr=1e-7, grad_norm=0.037

### Results (10K POC, Easy, 95% CI)

| Metric | Frozen | Adapted | Delta | Sig? |
|---|---|---|---|---|
| Prec@K | 36.1 ±0.6 | 14.3 ±0.3 | **-21.8pp** | YES |
| mAP@K | 0.278 ±0.006 | 0.080 ±0.002 | **-0.198** | YES |
| nDCG@K | 0.950 ±0.001 | 0.906 ±0.001 | **-0.045** | YES |
| Cycle@K | 76.0 ±0.8 | 75.5 ±0.8 | -0.5pp | NO |
| DimConsist@K | 36.1 ±0.6 | 35.7 ±0.6 | -0.4pp | NO |

### Diagnosis

λ=0.001 drift penalty (0.00047) is **1000x smaller** than JEPA loss (0.476) — effectively zero. Adapted Prec@K=14.3% collapsed to random level (12.2%). EWC literature uses λ=10²–10⁹ (arxiv 2505.05946). Val loss 1.648 >> train 0.476 = overfit/forgetting. Old POC run had JEPA loss=1.49 (broken ImageNet norm → training was ineffective → model stayed near frozen → false positive).

### Next

Sweep λ∈[1.0, 10.0, 100.0]. Keep LR=1e-5. Consider multi-epoch on 10K subset (arxiv 2406.14833: stability gap). Add early stopping on val loss.
