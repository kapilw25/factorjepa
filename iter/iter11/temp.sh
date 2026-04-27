## 1. Phase 1 — factor prep (m10 + m11)  >> COMPLETED
tmux new -s iter11_v3
./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml \
    2>&1 | tee logs/run_factor_prep_v3.log

## 2. Phase 2 — train 4 variants, ~30-60 h GPU
### 2)a) train - ExpLoRA >> RUNNING  
```bash
./scripts/run_train.sh \
    configs/train/explora.yaml \
    2>&1 | tee logs/run_train_explora_v10.log 
```


### 2)b) train - surgery_2stage_noDI
# instance 1 — surgery_2stage_noDI  >> COMPLETED       
CACHE_POLICY_ALL=2 ./scripts/run_train.sh \
    configs/train/surgery_2stage_noDI.yaml \
    2>&1 | tee logs/run_train_surgery_2stage_noDI_epoch15_v3.log #[probe] Trajectory across 78 stages:
                                                                                                                                                                           

### 2)c) train - surgery_3stage_DI
# instance 2 — surgery_3stage_DI  >> COMPLETED      
CACHE_POLICY_ALL=2 ./scripts/run_train.sh \
    configs/train/surgery_3stage_DI.yaml \
    2>&1 | tee logs/run_train_surgery_3stage_DI_epoch15_v2.log   #[probe] Trajectory across 78 stages: 


## 3. Phase 3 —  EVAL
### surgery variants >> COMPLETED
./scripts/run_eval.sh \
    configs/train/surgery_2stage_noDI.yaml \
    configs/train/surgery_3stage_DI.yaml \
    2>&1 | tee logs/run_eval_surgery_v2.log      