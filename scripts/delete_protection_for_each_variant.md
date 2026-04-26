rompts only fire when a cache # exists at known paths. Missing caches default to policy=1 (keep). 
Bypasses: # CACHE_POLICY_ALL=1|2 ./run_factor_prep.sh skips prompts; non-TTY stdin → 1

The script collects ALL cache-policy decisions UP FRONT (lines 63-185) before any compute starts. Three layers:                                                          
                                                                                                                                                                        
1. Variant detection (line 71-79): VARIANTS=() array built by checking each candidate student_encoder.pt — only variants with checkpoints are evaluated.                 
2. _check_and_prompt_any helper (line 102-126): for each (call-site × variant) combo, checks if any cached artifact exists at known paths. If yes:                       
- CACHE_POLICY_ALL=1|2 env → use that, skip prompt                                                                                                                     
- Non-TTY (! -t 0) → silent default to 1                                                                                                                               
- Otherwise → read -p "$key cache at $found [1=keep / 2=recompute] (Enter=1): " → store in POLICY[$key]                                                                
3. Dependency propagation (line 154-162): if user chose recompute upstream (m05), auto-invalidate downstream (m06, m08b). Then prints the full plan + asks "Proceed with 
this plan? [y/N]" once.                                                                                                                                                  
                                                                                                                                                                        
The chain then runs unattended overnight, passing --cache-policy "${POLICY[m05_$v]}" to each Python call so the .py-level input() prompt never fires. 