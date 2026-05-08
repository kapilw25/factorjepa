● 🚨 Task 1: NOT a software bug — it's an m10 quality-gate finding                                                           
                                                                                                                            
Disk vs index gap explained:                                                                   
                                                                                                                            
factor_manifest.json (9297 clips) says:                                                                                  
has_D_L=true:  9238 clips (potential)                                                                                    
has_D_A=true:  7702 clips (potential)               
has_D_I=true:  6860 clips (potential)                                                                                    
ALL 3 + tubes: 6771 clips (potential)                                                                                    
                                                                                                                            
BUT factor_manifest_quality.json says:                                                                                     
n_clips_with_D_L_quality:  98       ← only 98 PASS the m10 D_L blur completeness threshold                               
n_clips_with_D_A_quality:  75       ← only 75 PASS the m10 D_A signal-to-bg threshold                                    
n_clips_with_m10_quality:  9297     ← all measured, but very few pass                                                    
                                                                                                                            
build_factor_index reads from DISK (.npy files):                                                                           
D_L disk: 98 files → 89 in index (9 lost to safe_key/orphan)                                                             
D_A disk: 75 files → 68 in index                                                                                         
D_I disk: 3462 tube files (multi-tube per clip) → 62 clips with at least 1 tube                                          
                                                                                                                            
Final UNION: 91 clips (training pool for surgery)                                                                        
                                                                                                                            
Reality: m10 (SAM3 segmentation) quality-gates ~99% of clips out. Only 98 clips pass D_L blur threshold; only 75 pass D_A  
signal-to-bg threshold. Surgery's 91-clip pool = intersection of survivors. At FULL data scale, the same 91 applies because
m10 was already run on all 9297 clips and produced these quality-gated outputs.                                           
                                                                                                                            
This is a research-design choice (high-quality > high-N) baked into m10's preprocessing, NOT a software bug. Two options:  
- ✅ Accept it — this is what factor surgery is designed to be (small N, high quality interventions). v12 anchor showed    
+13.16 pp gain on this 91-clip pool.                                                                                       
- ❌ Relax m10 thresholds — would require re-running m10 + m11 (hours of GPU) to get a larger pool. Out of scope this iter.
                                                                                                                            
This is not your immediate fix concern. Now Task 2.
                                                                                                                            
---  