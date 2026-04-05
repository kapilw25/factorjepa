# Experiment Log — FactorJEPA Continual Pretraining

> Append-only. One entry per run. 6 fields per entry.
>
> | Field | Why |
> |---|---|
> | Date + git hash | Reproducibility — link code to results |
> | Train config (clips, epochs, frames, norm) | We lost 16f/64f info — never again |
> | Eval config (clips, frames, BS) | Same |
> | Results table with CI | Paper-ready numbers |
> | Diagnosis with citations | Why it failed/succeeded — prevents re-investigating |
> | Next steps | Decision trail for future self |

---

## Run 2026-04-05: λ=0.001, 1 epoch, 115K clips

**Config**: `configs/pretrain/vitg16_indian.yaml`
**Git**: `945d9c3`
**Train**: 115K clips, 1 epoch, BS=112, 16f, ImageNet norm=YES
**Eval**: 10K POC subset, 64f, BS=44

| Metric | Frozen | Adapted | Delta | Sig? |
|---|---|---|---|---|
| Prec@K | 36.1 ±0.6 | 14.3 ±0.3 | -21.8pp | YES |
| mAP@K | 0.278 ±0.006 | 0.080 ±0.002 | -0.198 | YES |
| nDCG@K | 0.950 ±0.001 | 0.906 ±0.001 | -0.045 | YES |
| Cycle@K | 76.0 ±0.8 | 75.5 ±0.8 | -0.5pp | NO |
| DimConsist@K | 36.1 ±0.6 | 35.7 ±0.6 | -0.4pp | NO |

**Diagnosis**: λ=0.001 is effectively zero (drift penalty 1000x smaller than JEPA loss). Model forgot spatial features — adapted Prec@K (14.3%) collapsed to random baseline level (12.2%). EWC literature uses λ=10²–10⁹ (arxiv 2505.05946). JEPA loss 0.476 confirms training was effective (unlike old POC run at 1.49 which was broken by missing ImageNet norm). Val loss 1.648 >> train loss 0.476 = classic overfit/forgetting.

**Next**: Sweep λ in [1.0, 10.0, 100.0]. Keep LR=1e-5. Consider multi-epoch on 10K subset instead of 1 epoch on 115K (arxiv 2406.14833: stability gap paper).
