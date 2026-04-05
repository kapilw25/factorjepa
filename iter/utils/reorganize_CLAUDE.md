                                
Current: 91 lines, 27 numbered rules, 4 hooks                                            
                                                                                        
Rules already enforced by hooks (shrink to 1-line reference each)                        
                                                                                        
┌──────────────┬────────────────────────┬─────────────────┬─────────────────────────┐    
│     Rule     │          Hook          │  Current lines  │      Can shrink to      │    
├──────────────┼────────────────────────┼─────────────────┼─────────────────────────┤    
│ 5 (pip       │                        │                 │ Already 1 line, just    │ 
│ install)     │ enforce-dev-rules.sh   │ 1 line          │ remove "ENFORCED"       │ 
│              │                        │                 │ boilerplate             │    
├──────────────┼────────────────────────┼─────────────────┼─────────────────────────┤ 
│ 6 (bare      │ enforce-dev-rules.sh   │ 1 line          │ Same                    │    
│ python3)     │                        │                 │                         │ 
├──────────────┼────────────────────────┼─────────────────┼─────────────────────────┤ 
│ 6.1 + 24     │                        │ 5 lines total   │                         │ 
│ (3-check     │ post-edit-lint.sh      │ (DUPLICATE)     │ Merge to 1 line         │    
│ gate)        │                        │                 │                         │ 
├──────────────┼────────────────────────┼─────────────────┼─────────────────────────┤    
│ 16 (fail     │ fail-hard-research.sh  │ 2 lines         │ 1 line                  │ 
│ hard)        │                        │                 │                         │ 
├──────────────┼────────────────────────┼─────────────────┼─────────────────────────┤ 
│ 66/git line  │ enforce-dev-rules.sh   │ 1 line          │ Already 1 line          │ 
├──────────────┼────────────────────────┼─────────────────┼─────────────────────────┤    
│ 13 (tqdm)    │ post-edit-lint.sh AST  │ 2 lines         │ 1 line                  │
│              │ check                  │                 │                         │    
└──────────────┴────────────────────────┴─────────────────┴─────────────────────────┘
                                                                                        
Savings: ~6 lines from dedup + boilerplate removal                                       

Rules that COULD become new hooks                                                        
                                        
┌──────────────────┬───────────────────────────────┬─────────────────────────────────┐   
│       Rule       │           New hook            │               How               │
├──────────────────┼───────────────────────────────┼─────────────────────────────────┤   
│ 23 (imports at   │ Add E402 to ruff in           │ ruff check --select             │
│ top)             │ post-edit-lint.sh             │ F821,F841,F811,E402 — 1 char    │
│                  │                               │ change                          │   
├──────────────────┼───────────────────────────────┼─────────────────────────────────┤
│ 27 (shellcheck)  │ New PostToolUse:Edit,Write on │ shellcheck "$FILE_PATH" — ~15   │   
│                  │  *.sh                         │ lines                           │
├──────────────────┼───────────────────────────────┼─────────────────────────────────┤   
│ 15 (no hardcoded │ Tricky — needs grep for       │ False positive heavy, keep      │
│  values)         │ patterns like = 10000         │ advisory                        │   
└──────────────────┴───────────────────────────────┴─────────────────────────────────┘
                                                                                        
Rules to extract to reference file (incident details)                                    

These 5 incidents bloat rules with WHY context. Move to iter/incidents.md, keep 1-line   
rule:                                   
                                                                                        
┌──────────┬─────────────────────────────────┬───────────────────────────────────────┐   
│   Rule   │          Incident text          │           Keep in CLAUDE.md           │
├──────────┼─────────────────────────────────┼───────────────────────────────────────┤   
│ 22       │ "6 of 10 val_loss points lost   │ "Training metrics MUST use JSONL with │
│          │ to OOM crash..."                │  fsync"                               │
├──────────┼─────────────────────────────────┼───────────────────────────────────────┤   
│ 25       │ "m06 bootstrap CI took 88       │ "Replace Python for-loops with NumPy  │
│          │ min... vectorized: <1 min"      │ vectorization for 1K+ items"          │   
├──────────┼─────────────────────────────────┼───────────────────────────────────────┤
│ 27       │ "ls glob with pipefail silently │ "Trace data flow, don't just grep for │   
│          │  killed..."                     │  flag existence"                      │   
├──────────┼─────────────────────────────────┼───────────────────────────────────────┤
│ RULES/58 │ "ImageNet normalization caused  │ "Eval MUST match frozen baseline      │   
│          │ -26% Prec@K"                    │ conditions exactly"                   │   
├──────────┼─────────────────────────────────┼───────────────────────────────────────┤
│ RULES/59 │ "torchcodec SIGSEGV on TAR      │ "Test FULL code path in REPL, not     │   
│          │ files"                          │ just import"                          │   
└──────────┴─────────────────────────────────┴───────────────────────────────────────┘
                                                                                        
Savings: ~15 lines                      

Rules to remove (Claude can infer from code)                                             

┌─────────────────────────────────────────┬──────────────────────────────────────────┐   
│                  Rule                   │              Why removable               │
├─────────────────────────────────────────┼──────────────────────────────────────────┤
│ 2 "Utils: @src/utils/"                  │ Obvious from directory structure         │
├─────────────────────────────────────────┼──────────────────────────────────────────┤
│ 4/4.1 docstring format                  │ Claude can see existing docstrings as    │   
│                                         │ examples                                 │   
├─────────────────────────────────────────┼──────────────────────────────────────────┤   
│ 7.2 "embeddings.paths.npy stores clip   │ Claude reads the code                    │   
│ keys"                                   │                                          │
├─────────────────────────────────────────┼──────────────────────────────────────────┤   
│ 7.3 "m05c reads embeddings.paths.npy"   │ Claude reads the code                    │
├─────────────────────────────────────────┼──────────────────────────────────────────┤   
│ 9/wandb details                         │ Claude can see wandb_utils.py            │
└─────────────────────────────────────────┴──────────────────────────────────────────┘   
                                        
Savings: ~8 lines                                                                        

Proposed reorganized structure                                                           
                                        
# FactorJEPA — CLAUDE.md
                                                                                        
## Environment
- GPU: RTX PRO 6000 Blackwell (96GB), PyTorch 2.12+cu128, CUDA 12.8                      
- Configs: `configs/pipeline.yaml` (runtime), `configs/pretrain/*.yaml` (training)       
- Modules: src/m00-m09. vjepa2 imports via `src/utils/vjepa2_imports.py` shim            
                                                                                        
## MUST Rules                                                                            
- FAIL LOUD — no CPU fallback, no `except: pass`, no `|| true`                           
- 95% CI on every metric (bootstrap, `utils/bootstrap.py`)                               
- Eval matches frozen baseline exactly (64f, ImageNet norm, same processor)              
- No hardcoded clip counts — use `configs/pipeline.yaml` or `get_total_clips()`          
- Epoch-based training, JSONL metrics with fsync, checkpoint resume                      
- Trace data flow after adding flags — grep for existence is NOT validation              
                                                                                        
## GPU Performance                                                                       
- GPU util ≥85%. Idle GPU = wasted money ($1.20/hr)                                      
- torch.compile + sdp_kernel nullcontext for adapted models                              
- FAISS-GPU, cuML-GPU for iterative algorithms. Never CPU in GPU scripts                 
- NumPy vectorization for 1K+ items. No Python for-loops on arrays                       
- Threading ONLY for I/O. Never ThreadPoolExecutor for tensor ops                        
                                                                                        
## Behavioral                                                                            
- Sr. Research Scientist: pros/cons, challenge when wrong, WEBSEARCH before recommending 
- Never "let the run complete" — checkpoints make interrupts cheap                       
- Test in REPL before restarting pipelines                                               
- Show grep proof when auditing. No "I checked everything" claims                        
- Git: message text only, never run git commands                                         
                                                                                        
## Hooks (deterministic enforcement)                                                     
- `enforce-dev-rules.sh` — blocks pip install, git, bare python3                         
- `post-edit-lint.sh` — py_compile + AST + ruff after every edit                         
- `fail-hard-research.sh` — blocks error-swallowing patterns                             
- `protect-checkpoints.sh` — blocks deletion of student_encoder.pt                       
                                                                                        
## Reference                                                                             
- Incidents & bug history: `iter/incidents.md`
- Training plan: `iter/iter7_training/plan_training.md`                                  
                                                                                        
Result: ~50 lines vs current 91. One-shot examples removed — Claude reads the code.      
                                                                                        
Pros and cons                                                                            
                                        
┌──────────────┬────────────────────────────────────┬────────────────────────────────┐   
│              │                Pro                 │              Con               │
├──────────────┼────────────────────────────────────┼────────────────────────────────┤   
│ Reorganize   │ 40% shorter → better rule          │ 30 min during active iteration │
│ now          │ adherence. Grouped → faster        │  cycle. Risk losing nuance     │
│              │ lookup. Deduped → no confusion     │                                │   
├──────────────┼────────────────────────────────────┼────────────────────────────────┤
│ Add          │                                    │ pipefail` class of bugs        │   
│ shellcheck   │ Catches `ls                        │ deterministically              │   
│ hook         │                                    │                                │
├──────────────┼────────────────────────────────────┼────────────────────────────────┤   
│ Add E402 to  │ Catches inline imports             │ Already enforced by rule 23 —  │
│ ruff         │ automatically                      │ low incremental value          │   
├──────────────┼────────────────────────────────────┼────────────────────────────────┤
│ Extract      │ 15 fewer lines in every            │ Incidents won't load unless    │   
│ incidents    │ conversation context               │ explicitly referenced          │   
├──────────────┼────────────────────────────────────┼────────────────────────────────┤
│ One-shot     │                                    │ Code examples go stale. Better │   
│ examples     │ Teaches by showing, not telling    │  to let Claude read actual     │   
│              │                                    │ code                           │
└──────────────┴────────────────────────────────────┴────────────────────────────────┘ 