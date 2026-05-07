  ┌─────────────────────────────────────┬────────────────────────┬────────────────────────────────────────┬────────────┐    
  │               Feature               │  run_factor_prep.sh    │      run_factor_prep_parallel.sh       │  Verdict   │ 
  │                                     │        (serial)        │                                        │            │    
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤    
  │ set -euo pipefail                   │ ✅                     │ ✅                                     │ parity     │    
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤ 
  │ cd <repo-root> + source venv +      │ ✅                     │ ✅                                     │ parity     │    
  │ mkdir -p logs                       │                        │                                        │            │    
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤    
  │ Mode flag (--SANITY/--POC/--FULL)   │ ✅                     │ ✅ (just added)                        │ parity     │ 
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤    
  │ LOCAL_DATA env-var override         │ ✅                     │ ✅                                     │ parity     │ 
  │ (default data/eval_10k_local)       │                        │                                        │            │ 
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤    
  │ LOCAL_DATA dir existence check      │ ✅                     │ ✅                                     │ parity     │    
  │ (FATAL if missing)                  │                        │                                        │            │    
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤    
  │ manifest.json existence check       │ implicit (m10 fails)   │ ✅ explicit                            │ parallel   │ 
  │                                     │                        │                                        │ better     │    
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤ 
  │ TRAIN_SUBSET env-var support        │ ✅ (lines 73, 82–86)   │ ❌ MISSING                             │ gap        │    
  │ (filter clip universe further)      │                        │                                        │            │ 
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤    
  │ Header banner showing LOCAL_DATA +  │ ✅                     │ partial (no TRAIN_SUBSET line)         │ gap        │ 
  │ TRAIN_SUBSET                        │                        │                                        │            │    
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤    
  │ Interactive cache-policy prompt     │ ✅ _check_and_prompt   │ ❌ MISSING (silently defaults to 2)    │ UX gap     │ 
  │ when no CACHE_POLICY_ALL + TTY      │                        │                                        │            │    
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤    
  │ Per-module cache policy (m10/m11    │ ✅                     │ ❌ shared single value (acceptable     │ minor      │ 
  │ separate)                           │                        │ since merge step runs anyway)          │            │    
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤    
  │ Dependency propagation (m10         │ ✅ explicit            │ ✅ implicit (same CACHE_POLICY)        │ parity     │ 
  │ recompute → m11 recompute)          │                        │                                        │            │    
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤ 
  │ Conditional --subset flag           │ ✅ ($SUBSET_FLAG       │ N/A — uses internally generated        │ by design  │    
  │ passthrough                         │ empty/2-token)         │ subsets                                │            │
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤    
  │ Variant-tagged per-step log         │ ✅                     │ ✅                                     │ parity     │    
  │ filenames                           │                        │                                        │            │
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤    
  │ Wall-time stamp on completion       │ ✅                     │ ✅                                     │ parity     │    
  ├─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┼────────────┤
  │ Resume after kill (m10's            │ ✅                     │ partial — see below ⚠️                  │ subtle gap │    
  │ processed_keys checkpoint)          │                        │                                        │            │    
  └─────────────────────────────────────┴────────────────────────┴────────────────────────────────────────┴────────────┘
                                                                                                                            
  Resume semantics — the one to actually worry about                                                                        
                                                                                    
  The parallel script's effective cache behavior:                                                                           
                                                                                                                            
  ┌────────────────────────┬──────────────────────────────┬─────────────────────────────────────────────────────────────┐
  │        Scenario        │    Serial behavior with      │          Parallel behavior with CACHE_POLICY_ALL=2          │   
  │                        │      CACHE_POLICY_ALL=2      │                                                             │
  ├────────────────────────┼──────────────────────────────┼─────────────────────────────────────────────────────────────┤   
  │ First run, empty disk  │ Wipe canonical, run fresh    │ No wipe needed; workers run on empty scratch                │   
  ├────────────────────────┼──────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ Re-run after killed    │ Wipe canonical, restart from │ Wipes worker scratch → restarts each worker from 0 of its   │   
  │ partial                │  0 (worker lost)             │ 2370-clip slice → loses up to ~25-30 min of partial worker  │
  │                        │                              │ progress                                                    │   
  ├────────────────────────┼──────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ Re-run with            │ Resume m10 from              │ Resume each worker from its own scratch checkpoint          │   
  │ CACHE_POLICY_ALL=1     │ processed_keys               │ (correct)                                                   │
  └────────────────────────┴──────────────────────────────┴─────────────────────────────────────────────────────────────┘
                                                                                                                            
  Net: parallel is OK on re-run if you remember CACHE_POLICY_ALL=1, but the default =2 semantic of "fresh start" is         
  misleading because canonical is never wiped, only worker scratch is.                                                      
                                                                                                                            
  Three additions to bring parallel to parity                    
                                                                                    
  ┌─────┬───────────────────────────────────────────────────────────┬─────────────────────────┬─────────────────────────┐
  │  #  │                            Add                            │           Why           │           LoC           │   
  ├─────┼───────────────────────────────────────────────────────────┼─────────────────────────┼─────────────────────────┤
  │     │ TRAIN_SUBSET env-var → if set, splitter reads keys from   │ Parity with serial;     │                         │   
  │ 1   │ TRAIN_SUBSET (a {"clip_keys": []} JSON) instead of        │ lets user filter to     │ ~5 in shell + ~5 in     │
  │     │ manifest.json["saved_keys"].                              │ e.g. only chennai       │ m10_split_subset.py     │   
  │     │                                                           │ walking clips.          │                         │   
  ├─────┼───────────────────────────────────────────────────────────┼─────────────────────────┼─────────────────────────┤
  │ 2   │ Header banner: print TRAIN_SUBSET status (set/unset) like │ UX parity.              │ 4                       │   
  │     │  serial does.                                             │                         │                         │
  ├─────┼───────────────────────────────────────────────────────────┼─────────────────────────┼─────────────────────────┤   
  │     │ Interactive cache-policy prompt (or at minimum a "no      │ Prevents accidental     │                         │
  │ 3   │ CACHE_POLICY_ALL set, defaulting to 2 — Ctrl-C in 5s if   │ scratch wipe on re-run. │ ~20                     │   
  │     │ wrong" warning). Better: borrow _check_and_prompt style.  │                         │                         │   
  └─────┴───────────────────────────────────────────────────────────┴─────────────────────────┴─────────────────────────┘
                                                                                                                            
  Want me to add all 3? Total ~35 LoC, all in the shell script + ~5 in m10_split_subset.py. Doesn't touch the merge or m11
  logic, so SANITY smoke test (already in plan as #182) still validates the new code paths.  