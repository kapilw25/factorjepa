
## 2. Phase 2 — train 2 variants (FULL, ~14 GPU-h sequential)

```bash
tmux new -s iter12_train

CACHE_POLICY_ALL=2 ./scripts/run_train.sh \
    configs/train/surgery_2stage_noDI_multitask.yaml \
    configs/train/surgery_3stage_DI_multitask.yaml \
    2>&1 | tee logs/run_train_iter12_multitask_v3.log
```

## 3. Phase 3 — EVAL

### 3.1 Evaluate 5 variants (3 standard from iter11 v3 + 2 new multitask)

```bash
./scripts/run_eval.sh \
    configs/train/explora.yaml \
    configs/train/surgery_2stage_noDI.yaml \
    configs/train/surgery_3stage_DI.yaml \
    configs/train/surgery_2stage_noDI_multitask.yaml \
    configs/train/surgery_3stage_DI_multitask.yaml \
    2>&1 | tee logs/run_eval_iter12_multitask_v1.log
```
