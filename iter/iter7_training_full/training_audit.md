---                                                                                                                                                                                        
name: audit-training
description: Audit GPU training scripts against reference implementations and config files. Catches orphan functions, unused config keys, missing validation loops, and deviations from    
Meta's official V-JEPA training code.                                                                                                                                                      
disable-model-invocation: true                                                                                                                                                             
allowed-tools: Read, Grep, Glob, Bash                                                                                                                                                      
argument-hint: [file-path, e.g. src/m09_pretrain.py]                                                                                                                                       
---                                                                                                                                                                                        

# Training Script Audit

Audit the specified training script (default: `src/m09_pretrain.py`) for completeness and correctness. Run BEFORE any long GPU training job.

## Checks (run ALL, report PASS/FAIL with evidence)

### 1. Orphan Functions
- List every `def func_name(` in the file
- For each function, grep for `func_name(` calls in the SAME file
- `main()` and functions called by argparse are exempt
- FAIL if any function is defined but never called (like `run_validation()` was)

### 2. Config-Code Alignment
- Read the YAML config file (`configs/pretrain/vitg16_indian.yaml`)
- For every leaf key (e.g., `validation.interval_steps`, `drift_control.lambda_reg`), grep for that key name in the Python file
- FAIL if any config key is never referenced in the code (dead config = dead feature)

### 3. Reference Implementation Diff
- Read Meta's official training loop: `deps/vjepa2/app/vjepa/train.py`
- Compare against our `src/m09_pretrain.py` for these critical components:
- Loss function (L1 vs MSE, loss_exp)
- EMA update (momentum, _foreach_mul_ pattern)
- Mask generation (MaskCollator vs _MaskGenerator)
- Teacher forward (all tokens, no grad, layer_norm)
- Student forward (masked tokens only)
- Optimizer (AdamW, param groups, weight decay on bias/norm)
- FAIL if any component differs without documented justification

### 4. Training Loop Completeness
- Verify ALL of these exist in the training loop (not just defined somewhere):
- [ ] Forward pass (student + teacher)
- [ ] Loss computation
- [ ] Backward pass
- [ ] Optimizer step
- [ ] EMA teacher update
- [ ] Learning rate scheduler step
- [ ] Gradient clipping
- [ ] **Validation loop** (called periodically, not just defined)
- [ ] Checkpoint save (periodic + best)
- [ ] WandB logging (train metrics + val metrics)
- [ ] tqdm progress bar
- FAIL if any component is missing from the actual loop

### 5. WandB Metric Coverage
- List every `log_metrics(wb_run, {...})` call
- Check that both train AND val metrics are logged
- Check that loss, lr, grad_norm, throughput are logged
- FAIL if only train metrics logged (no val/* prefix metrics)

### 6. Data Integrity
- Verify train/val split exists and val set is actually used
- Verify no data leakage (val clips never enter training producer)
- Verify checkpoint resume doesn't corrupt train/val split

## Output Format

```
AUDIT: src/m09_pretrain.py
[1] Orphan Functions:     PASS/FAIL (list orphans)
[2] Config Alignment:     PASS/FAIL (list unused keys)
[3] Reference Diff:       PASS/FAIL (list deviations)
[4] Loop Completeness:    PASS/FAIL (list missing components)
[5] WandB Coverage:       PASS/FAIL (list missing metrics)
[6] Data Integrity:       PASS/FAIL (list issues)

TOTAL: X/6 passed. Y critical issues found.