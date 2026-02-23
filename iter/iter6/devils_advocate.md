⏺ Confirmed: sys is only imported inside 3 specific functions (lines 155, 189, 346), but sys.exit(1) on line 562 is inside check_output_exists() which has NO local import sys. CRITICAL bug
confirmed.                                                                                                                                                                               
                                                                                                                                                                                        
Now let me compile the full findings and start fixing. Here's the consolidated report:                                                                                                    
                                                                                                                                                                                        
---                                                                                                                                                                                       
Full Pipeline Audit: Consolidated Findings                                                                                                                                                
                                                                                                                                                                                        
CRITICAL (will crash)                                                                                                                                                                     
                                                                                                                                                                                        
┌─────┬───────────┬───────┬───────────────────────────────────────────────────────────────────────────────────┐                                                                           
│  #  │   File    │ Lines │                                       Issue                                       │
├─────┼───────────┼───────┼───────────────────────────────────────────────────────────────────────────────────┤
│ C1  │ config.py │ 562   │ sys.exit(1) in check_output_exists() but sys never imported → NameError on Ctrl+C │
└─────┴───────────┴───────┴───────────────────────────────────────────────────────────────────────────────────┘

HIGH (wrong results or major perf loss)

┌─────┬──────────────────────┬──────────┬───────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
│  #  │         File         │  Lines   │   Category    │                                         Issue                                         │
├─────┼──────────────────────┼──────────┼───────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
│ H1  │ m06_faiss_metrics.py │ 639-645  │ Data Loss     │ No key-matching between embeddings & tags → silently wrong metrics on misaligned data │
├─────┼──────────────────────┼──────────┼───────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
│ H2  │ m05_vjepa_embed.py   │ 228, 238 │ GPU Underutil │ Single-threaded video decoding starves GPU (m04 uses ThreadPoolExecutor)              │
├─────┼──────────────────────┼──────────┼───────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
│ H3  │ m05_vjepa_embed.py   │ 496-497  │ GPU Idle      │ save_checkpoint blocks GPU with growing np.stack + disk I/O every 500 clips           │
├─────┼──────────────────────┼──────────┼───────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
│ H4  │ m05_vjepa_embed.py   │ 202      │ GPU Idle      │ Resume always streams from clip 0, re-iterating through all processed clips           │
└─────┴──────────────────────┴──────────┴───────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

MEDIUM (noticeable perf/correctness issues)

┌─────┬───────────────────────┬─────────┬────────────────┬───────────────────────────────────────────────────────────────────────────┐
│  #  │         File          │  Lines  │    Category    │                                   Issue                                   │
├─────┼───────────────────────┼─────────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ M1  │ m04b_vlm_select.py    │ 95-98   │ Data Loss      │ Majority vote assumes identical clip ordering without key validation      │
├─────┼───────────────────────┼─────────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ M2  │ m07_umap_plot.py      │ 274-277 │ Visual         │ kNN grid always shows placeholders (HF keys ≠ local paths)                │
├─────┼───────────────────────┼─────────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ M3  │ m01_download.py       │ 247-251 │ Reporting      │ Skip/ok counting broken; str(True) never contains "SKIP"                  │
├─────┼───────────────────────┼─────────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ M4  │ config.py             │ 341-376 │ Dead code trap │ load_embeddings_and_tags() not POC-aware; wrong files in POC mode if used │
├─────┼───────────────────────┼─────────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ M5  │ m00c_sample_subset.py │ 183     │ Crash          │ IndexError on empty sample input                                          │
├─────┼───────────────────────┼─────────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ M6  │ m06_faiss_metrics.py  │ 186-192 │ GPU Underutil  │ Overlap@K builds 2 CPU FAISS indices; wastes time on GPU server           │
└─────┴───────────────────────┴─────────┴────────────────┴───────────────────────────────────────────────────────────────────────────┘

LOW (minor/cosmetic)

┌─────┬────────────────────────────┬───────────────────────────────────────────┐
│  #  │            File            │                   Issue                   │
├─────┼────────────────────────────┼───────────────────────────────────────────┤
│ L1  │ config.py:101              │ QWEN_MODEL_ID dead code                   │
├─────┼────────────────────────────┼───────────────────────────────────────────┤
│ L2  │ config.py:30-34            │ Split repo constants dead code            │
├─────┼────────────────────────────┼───────────────────────────────────────────┤
│ L3  │ config.py:459              │ Docstring says "m04's" should say "m05's" │
├─────┼────────────────────────────┼───────────────────────────────────────────┤
│ L4  │ hf_utils.py:2              │ Docstring references old m04_qwen_tag.py  │
├─────┼────────────────────────────┼───────────────────────────────────────────┤
│ L5  │ config.py:122-123          │ UMAP_PLOT_PNG/PDF unused constants        │
├─────┼────────────────────────────┼───────────────────────────────────────────┤
│ L6  │ m00b_fetch_durations.py:94 │ Variable shadows function parameter       │
├─────┼────────────────────────────┼───────────────────────────────────────────┤
│ L7  │ m06_faiss_metrics.py:269   │ mAP@K uses non-standard normalization     │
└─────┴────────────────────────────┴───────────────────────────────────────────┘

WebSearch: Additional failure modes to check

From web research, the key categories beyond OOM/underutil/idle are:
- Data starvation (46% of GPU underutil per Microsoft study) — found in H2
- VRAM memory leaks across long runs — m04 already mitigates via orchestrator/worker restart pattern
- Numerical instability with fp16 — V-JEPA uses flash-attn-2 which handles fp16 correctly
- Checkpoint corruption — found and already fixed (np.savez naming bug)

---