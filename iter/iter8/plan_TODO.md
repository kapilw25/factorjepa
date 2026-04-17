# TODO — iter8

> **Final GOAL: Surgery > ExPLoRA > Frozen on Prec@K, 115K clips.**
> **Immediate GOAL: Surgery > ExPLoRA > Frozen on Prec@K, 1K clips from @data/val_1k_local/manifest.json**
> **m10/m11 Goal = maximize D_A/D_L/D_I accuracy for Prec@K**
> **Deadline: NeurIPS May 04. Budget: ~38h remaining (22 days x 2h/day minus 6h spent).**
> **If surgery doesn't improve:** `iter/utils/literarure_survey.md` — 24 JEPA variants, 3 fallback techniques.

---

## ✅ DONE: Grounded-SAM Pivot (2026-04-14, on GPU directly)

**Architecture (locked in):**
```
Grounded-SAM Path D:  fixed 17-cat agent taxonomy → Grounding DINO (text → boxes on frame 0)
                      → SAM 3.1 add_prompt(text=cat, boxes=DINO_xywh_norm, box_labels=[1]*N)
                      → SAM 3.1 propagates masks across all 16 frames (text drives tracking,
                        boxes refine frame 0). Per-category sessions preserve obj_id ranges
                        (offset += 100) for D_I cross-category mining.
```

**Verified on SANITY v5 (20 clips, 24GB RTX PRO 4000 Blackwell):**
- 12/20 clips with detected agents (8 truly-empty Goa/monument scenes correctly skipped)
- 82 total agent detections, mean mask confidence 0.93
- 39 interaction tubes from 9/20 clips (vs 0 with text-only or boxes-only)
- Per-clip 2x2 verify grids show clean D_L blur, D_A isolation, D_I crops
- m10 quality gate PASS, m11 manifest produced for all 20 clips

**Done tasks:**
- ✅ Modified `m10_sam_segment.py`: DINO box detection + SAM 3.1 text+boxes hybrid (Path D)
- ✅ Added `iopath`, `ftfy` to `requirements_gpu.txt` (SAM3 `--no-deps` undeclared deps)
- ✅ Added `load_dotenv()` to m10 (HF_HOME, HF_TOKEN propagation)
- ✅ Fixed transformers 4.57 API renames (`box_threshold`→`threshold`, `labels`→`text_labels`)
- ✅ Fixed SAM 3.1 box format (xyxy→normalized xywh, paired box_labels=[1]*N)
- ✅ Tuned thresholds Option C: DINO box=0.15, text=0.12; m11 min_agent_area_pct=0.003
- ✅ Updated `runbook.md` paths: `outputs/sanity/factors/` → `outputs/sanity/m10_sam_segment/` etc.
- ✅ Fixed taxonomy: 17 agent categories in `ch11_surgery.yaml` (not VLM tags) for accuracy
- ✅ Added step [9/9] to `setup_env_uv.sh` to pre-cache Grounding DINO weights

**Documented in `errors_N_fixes.md` entries #18-27.**

---

## 🏗️ Current m10/m11 Architecture (2026-04-14, Level 2)

**Pipeline:** Grounded-SAM (DINO detection + SAM 3.1 mask refinement + tracking), multi-anchor re-seed.

| Stage | Tool | Frames | Purpose |
|---|---|---|---|
| Detection | Grounding DINO base (17-cat taxonomy) | Anchors `[0, 4, 8, 12]` | Open-vocab text→box on frame 0 of each 4-frame segment |
| Refinement + tracking | SAM 3.1 multiplex, text+boxes hybrid | Per anchor, segment `[a..a+3]` | `add_prompt(frame_index=anchor, text=category, boxes=...)` → propagate within 4 frames |
| D_I mining | `mine_interactions()` geometry | Full 16 frames | Centroid pair runs ≥4 frames within 20% frame width |

**Why multi-anchor (Level 2)**: single-anchor drifted 25%→5% by mid-frame; 4 anchors cap drift at ≤2 frames. 4× runtime for +334% agents / +51% D_I. See `errors_N_fixes.md` #32.

**D_I tube builder (2026-04-14)**: m10 now saves `per_object_bboxes_json` (~5 KB/clip); m11 prefers tight union-bbox crops over fixed 30% centroid squares (graceful fallback for legacy .npz).

### 📊 POC dense100 measurements — Level 1 → Level 2 → **v2_HF + bbox-tubes** (current, 2026-04-15)

| Metric | Level 1 (1 anchor, raw sam3) | Level 2 (4 anchors, raw sam3) | **v2_HF (4 anchors, HF Sam3Tracker + bbox tubes)** |
|---|---:|---:|---:|
| m10 throughput | 12.86 s/clip | 46.38 s/clip | **11.02 s/clip** ✅ |
| m10 total agents | 1286 | 5581 | **6146** |
| m10 D_I interactions | 1759 | 2659 | **8723** |
| m10 mean pixel_ratio | 17.42% | 21.66% | 18.61% (tighter masks) |
| m11 D_L / D_A / D_I present | — | 100 / 93 / 88 | **100 / 94 / 91** |
| m11 total tubes | — | 2659 (fixed 30 % squares) | **8723 (5659 unique bbox shapes)** |
| m11 median tubes/clip | — | 21.5 | **65** |
| 115K ETA (24GB, m10 only) | ~15 days | ~61 days | **~14.7 days** |
| 115K ETA (96GB, batch ×4) | — | — | **~3.7 days** |

Sources: `logs/m10_v2HF_dense100_probe5_v5.log`, `logs/m11_dense100_level2_v5.log` (m11 over v2_HF masks).
Verdict: Path B achieved 4.21× speedup AND +228 % D_I tubes AND tighter agent masks — a clean Pareto win.

**Decision log:**
- ✅ 17-category agent taxonomy in `configs/train/ch11_surgery.yaml > factor_datasets.grounding_dino.agent_taxonomy` (fixed, not per-clip VLM tags) — accuracy-first for D_L/D_A/D_I
- ✅ Option C thresholds: DINO `box=0.15, text=0.12` (aggressive recall), m11 `min_agent_area_pct=0.003`
- ✅ Path D text+boxes hybrid in SAM 3.1 `add_prompt` (not boxes-only which drops tracking)
- ✅ Box clamp before xywh-normalization (errors #25, #28)
- ✅ Guards on empty add_prompt output (#30) + SAM 3.1 state inconsistency (#31)
- ✅ `verify_or_skip` completeness check (#29) — partial output now resumes instead of skipping

---

## 🔥 Active (Phase 1 ✅ GREEN → Phase 2 🎯 1K POC NEXT)

- ✅ Step A (SANITY 20-clip + POC 100-dense 2026-04-17): m10 Grounded-SAM — quality gate PASS, 6141 agents, 8712 interactions, 6.13 s/clip on 96GB
- ✅ Step B (SANITY 20-clip + POC 100-dense 2026-04-17): m11 factor datasets — 91/100 D_I clips, 8712 tubes, 47 s with 32-worker ProcessPool (5.7× speedup)
- ✅ Step C (m05 frozen V-JEPA 2.1, 2026-04-17): 100 clips × 1664-dim in 423 s. `torch.compile` + bf16 + RoPE cast (#44/#59 durable) all working.
- ✅ Step D.1 (m09c Surgery SANITY, 2026-04-17): all 3 stages PASS — 0.4870 / 0.4901 / **0.4806** (first ever Stage 3). 96GB resolved #58 for free.
- 🐛 Step D.2 v1 (m09c Surgery POC 100-dense, 2026-04-17 first run): completed in 60 s — revealed **#60** `max_epochs.poc: 1` → only 3 optimizer steps total. Fixed → max_epochs.poc: 100.
- 🐛 Step D.2 v2 (m09c Surgery POC 100-dense, 2026-04-17 second run): completed in ~95 min — revealed **#61** `warmup_steps: 200 > stage_steps: 99` → LR never reached target, loss 0.50→0.476 warmup-truncated. Fixed → `warmup_pct: 0.20` auto-scaling.
- 🚚 100-dense tier retired. 3200 visits/clip = unpublishable overfitting pressure. Moving to 1K val_1k POC tier.
- 🎯 **Step D.2 v3 (1K val_1k POC, NEXT)**: both bug fixes in place + `max_epochs.poc: 20` for ~2.7 h wall. Real training signal.
- ⬜ Step D.3 (m05 re-embed on 1K surgical, ~70 min)
- ⬜ Step D.4 (m06 Prec@K frozen vs surgical — decision gate, ~5 min)
- ⬜ Step E.1 (m09b ExPLoRA SANITY, ~10 min) — run after D.4 regardless of gate result (cheap code smoke test)
- ⬜ Step E.2 (m09b ExPLoRA POC 1K, ~2 h) — CONDITIONAL on D.4 showing Surgery > Frozen
- ⬜ Step E.3 (m05 ExPLoRA re-embed + m06 Prec@K) — completes the 3-arm comparison

---

### ✅ m09c SANITY training — RESOLVED 2026-04-17 on 96GB Blackwell

**Iterations:** v0 → v7 on 24GB, v8-hw on 96GB. Errors & fixes catalogued in `errors_N_fixes.md` #50-#58.

| Version | Furthest point reached | Blocker | Fixed by |
|---|---|---|---|
| v0 | `build_model` | `KeyError 'src.models.predictor'` | #50 (vjepa2_imports finally-block restores all saved_modules) |
| v1 | `build_model:125` | `KeyError 'patch_size'` on cfg["data"] | #51 (sed `data_cfg[X]` → `model_cfg[X]` for crop_size/patch_size/tubelet_size) |
| v2 | `train_surgery:351` | `TypeError 'int' not subscriptable` on max_epochs | #52 (drop redundant `[mode_key]` — merge already flattened) |
| v3 | Stage 1 step 0 | OOM at first forward (2.25 GiB need, 1.87 free on 24GB) | #53 (AdaptiveBatchSizer + `_train_step_grad_accum` wired + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) |
| v4 | Stage 1 summary | `UnboundLocalError jepa_val` (step OOMed → loop ended before any value assigned) | #54 (pre-init loss vars per-stage before inner for-loop) |
| v5 | All 3 stages printed "complete" | Silent fail: 0 successful steps, exported unmodified student | #55 (within-step retry on OOM + fail-hard when sizer at min) |
| v6 | Stages 1+2 ✅ (loss=0.4841, 0.4874), Stage 3 OOM | Stage 3: 36/48 trainable blocks — fp32 master (5.5 GB) + 8-bit m1/m2 (3 GB) + model + activations overflowed 24 GB | #56 (grad checkpointing + bnb AdamW8bit) + #57 (mode-gated yaml: savers ON for SANITY, OFF for POC/FULL) |
| v7 | Stages 1+2 ✅, Stage 3 OOM at min sub-batch=1 | fp32 master + 8-bit m1/m2 + activation spike exceeded 24GB even with PagedAdamW8bit | #58 (inter-stage cleanup + PagedAdamW8bit — helped but didn't close the gap on 24GB) |
| ✅ **v8-hw (2026-04-17 noon)** | **ALL 3 STAGES PASS on 96GB** (Stage 1=0.4870, Stage 2=0.4901, **Stage 3=0.4806**), student_encoder.pt exported | — (resolved by 96GB hardware migration; post-cleanup VRAM 19.9 GB / 102 GB after Stage 2) | 🏁 Hardware upgrade; "v8 teacher CPU offload" patch from #58's follow-up plan NOT needed. |
| 🐛 **v9 (2026-04-17 dinner-time POC)** | D.2 POC 100-dense "SURGERY COMPLETE" in 60 s — but only 3 optimizer steps total 😱 | `max_epochs.poc: 1` (yaml comment "1 epoch per stage" ≠ code "1 epoch total"). `stage_steps = int(3 × 0.33) = 0` → clamped to 1/stage. Silent near-no-op. | #60 — bumped `max_epochs.poc: 1 → 100`, yaml comment corrected. |
| 🐛 **v10 (2026-04-17 late POC)** | D.2 POC 100-dense 300 steps completed — loss dropped only 0.50 → 0.476 (∆ −5.4 %) 📉 | `warmup_steps: 200` per stage > `stage_steps: 99` → LR never reached target (~50 % max). Fresh scheduler per stage restarts warmup from 0. | #61 — replaced fixed `warmup_steps` with `warmup_pct: 0.20` in yaml, auto-scales to 19/41/718 for 100/1K/115K. |
| 🎯 **v11 (1K val_1k POC, NEXT)** | (pending) | — | — |

**Closure:** SANITY loop resolved via hardware (v7→v8). POC loop debugged via config-fix cascade (#60 → #61). Real Prec@K signal will come from v11 at 1K scale.

### 🛑 v7 Stage 3 OOM — detailed root cause

**What v7 did accomplish:** `PagedAdamW8bit` (unified-memory CPU paging) wired in via `#58`; inter-stage optimizer cleanup (`None`-ref + `empty_cache` + `ipc_collect`) working. Stage 2 → Stage 3 transition log shows `17.7 GB used / 25.2 GB total after releasing Stage 2 state` — cleanup IS happening correctly. Stage 3's optimizer built successfully. Forward at sub-batch=1 immediately OOMed.

**Memory accounting for Stage 3 on 24 GB (actual, from v7 log):**

| Component | Size | Cumulative |
|---|---:|---:|
| CUDA context + PyTorch reserved | ~2.0 GB | 2.0 GB |
| Student fp16 (48 blocks × 2B params) | 3.7 GB | 5.7 GB |
| Teacher fp16 (frozen EMA copy) | 3.7 GB | 9.4 GB |
| Predictor fp16 (60M) | 0.12 GB | 9.5 GB |
| Mixed-precision buffers, mask generators, etc. | ~0.5 GB | 10.0 GB |
| **Post-cleanup baseline ← matches log "17.7 GB"** (7.7 GB extra accounted for by partial optimizer pre-allocation) | — | 17.7 GB |
| Stage 3 optimizer: fp32 master (1.38B × 4 bytes) | 5.5 GB | — |
| Stage 3 optimizer: 8-bit m1+m2 (quantized) | ~3.0 GB | — |
| PagedAdamW8bit pages SOME of above to CPU RAM | -3 to -5 GB | ~20 GB after build |
| **Forward spike at sub-batch=1** (activations + grads + intermediate) | **+5-6 GB** | **~25-26 GB → OOM** |

**Why PagedAdamW8bit alone wasn't enough:** paging is *reactive* (pages on pressure, not proactively). The forward-pass spike is fast — allocator requests activation tensors, there's no free slot, it returns OOM *before* paging can swap anything out. Paging helps steady-state memory but not burst peaks.

### ✅ Proposed fix for v8 — teacher CPU offload

**Idea:** Teacher is 3.7 GB permanently resident on GPU but only *used* during a single `torch.no_grad()` forward per sub-batch (inside `_train_step_grad_accum`). For 95%+ of each step the teacher sits idle consuming premium GPU memory. Move teacher to CPU by default, swap to GPU only for that one forward call, move back to CPU after.

**Expected memory freed:** 3.7 GB on GPU permanently.
**Expected throughput hit:** ~200-500ms per sub-batch for 3.7 GB PCIe transfer (× 2: to-GPU then back). At SANITY's scale (1 step per stage), ~1 second added per step total. Negligible.
**Accuracy impact:** ZERO — teacher runs at full fp16 precision on GPU during its forward; the only change is residency between calls.

**Implementation sketch** (surgery-only per scope discipline):
1. Add mode-gated yaml: `teacher_offload: {sanity: true, poc: false, full: false}` in `ch11_surgery.yaml` + flatten in `m09c_surgery.merge_config_with_args`.
2. In `_train_step_grad_accum` (or a surgery-only wrapper around it), if offload flag is True:
   ```python
   teacher = teacher.to(device, non_blocking=True)
   with torch.no_grad():
       h = teacher(bc)
   teacher = teacher.to("cpu", non_blocking=True)
   torch.cuda.synchronize()  # ensure copy-out completes before student forward
   ```
3. Store teacher on CPU after initial build in m09c `build_model` (conditional).
4. EMA update (`update_teacher_ema`) also needs teacher on GPU briefly — swap pattern same as above.

**Gold-standard reference:** HF Accelerate's `cpu_offload_with_hook` pattern, DeepSpeed ZeRO-Infinity's weight offloading, FAIR vissl's `param_offload_to_cpu`. All use the same "move to GPU on use, back on done" semantics.

**Expected v8 Stage 3 memory:** 17.7 GB - 3.7 GB teacher = 14.0 GB post-cleanup → +8.5 GB Stage 3 optimizer (partially paged) ≈ 20 GB → +5-6 GB forward spike ≈ 25-26 GB. Still tight. **May need a second fix.**

### 📋 Fallback fixes if teacher offload alone doesn't fit v8

Ordered by preference (least invasive first):

1. **Reduce teacher to fp32→bf16 master only + fp16 weights** (already in mixed precision, marginal)
2. **Disable teacher hierarchical output (4-level deep supervision) for SANITY Stage 3** — saves ~1.5 GB in teacher forward activations. Config flag `cfg[model][n_output_distillation]=1` for SANITY. Changes training loss slightly but SANITY is code-only.
3. **Reduce SANITY `num_frames` from 16 → 8** — halves token count, halves activation memory. Add mode-gate to `data.num_frames`. Clean yaml-only change.
4. **Predictor CPU offload** (predictor is small, 0.12 GB — low ROI)
5. **Gradient accumulation at sub-micro-batch level** — split sub-batch=1 forward into time-sliced chunks. Non-trivial code change.

### ✅ Status at 2026-04-17 (RESOLVED on 96 GB)

- All SANITY code paths validated end-to-end on 96 GB Blackwell: Stage 1 loss=0.4870, Stage 2 loss=0.4901, **Stage 3 loss=0.4806** (first successful measurement — first ever Stage 3 completion).
- Stage 3 peak post-cleanup VRAM: 19.9 GB / 102 GB — ~80 GB of headroom. No OOM, no fallback needed.
- **v8 teacher-CPU-offload patch was NOT landed** — prediction from 2026-04-15 ("may not fit even with offload") was superseded by the cheaper option (just run on 96 GB). Leaves code clean (no offload complexity to maintain) and matches plan_training's SANITY→POC hardware progression.
- **POC D.2 unblocked** — all savers will auto-flip OFF (mode-gated yaml #57) for clean fp32 AdamW training, the research-quality recipe for the Prec@K comparison.

---

## 🚧 Phase 2: POC (96GB GPU, after SANITY passes on 24GB)

- ⬜ POC: `train_explora.sh --POC` + `train_surgery.sh --POC`
- ⬜ 🎯 Decision gate: compare Prec@K frozen vs ExPLoRA vs surgical

| POC result | Next action |
|---|---|
| Surgery > Frozen (significant) | Phase 3: scale to 115K |
| Surgery > Frozen (within CI) | More epochs or tuning |
| Surgery = Frozen | Debug factor quality, try different stage order |
| Surgery < Frozen | Pivot: ExPLoRA-only paper, or temporal projection diagnostic |
| ExPLoRA works, surgery doesn't | Submit ExPLoRA + factor analysis paper (0h pivot) |
| Nothing works | Frozen 2.1 + temporal projection diagnostic paper (2h pivot) |

---

## ⏩ SPEEDUP (TODO) for FULL (115K clips) mode

Central registry for speedups across `src/m*.py` + `scripts/*.sh`. Add one row per option. "Status" = ✓ done / 🔬 tested, failed / ⬜ pending.

Measured baseline on POC dense100 (46 s/clip with v5 forward-only): **115K naive projection = ~61 days on 24GB GPU**. Path A alone is NOT sufficient for FULL — Path B (or equivalent) is mandatory.

| Module | Path | Effort | Speedup | 115K ETA (24GB) | Status |
|---|---|---|---|---|---|
| m10 | **A. `propagation_direction="forward"`** (skip backward SAM3 call) | done | 1.83× measured | ~61 days | ✓ #35 (unblocks POC, not FULL) |
| m10 | **B. HF `Sam3TrackerVideoModel` (replaced `m10_sam_segment.py`)** — requires `transformers==5.5.4` | done | **4.21× measured** | **~14.7 days** | ✅ #36-#40 validated 2026-04-15 on dense100 |
| m10 | B+96GB. Path B + larger batch on 96GB GPU | +0h | ~4× on top | **~3.7 days** | ⬜ (preferred for FULL) |
| m10 | B'. P-3a probe: `Sam3VideoModel` text-only (stripped from m10 code, kept in git history) | post-paper | +dropping DINO ~2× | ~1.5 days | ⬜ (backlog — not on critical path) |
| m10 | C. Streaming mode (HF only) — disables hotstart heuristics (quality risk) | ~3h | 10× | ~6 days | ⬜ (not recommended) |
| m10 | D. Density-filter FULL to ~30-40K multi-agent clips only | ~30min | — | ~2-4 days with B+96GB | ⬜ (paper-valid if stratified) |
| m10 | — `max_frame_num_to_track=3` in raw sam3 pkg | tried | would be 10× | — | 🔬 #33/#35 (SAM3 bug: empty tensor, reverted) |

**Default plan**: POC dense100 finishes with Path A (validates Level 2 quality). Then implement Path B BEFORE FULL — at current speed, 115K is ~61 days on 24GB GPU. Path B+96GB drops FULL to ~1.5 days, which fits the deadline.

---

## 🔬 Phase 3: Ablations (if POC positive)

> **Paper-grade metric for every ablation below: downstream Prec@K from surgery training.** Upstream proxies (tube area, mask confidence, concept_recall) are useful for debugging but only Prec@K validates any upstream change. Bootstrap 95% CI mandatory (`utils/bootstrap.py`).

| Ablation | What it shows | GPU time |
|---|---|---|
| **A1: Stage contribution** | Stage 1 only, 1+2, 1+2+3 — does each stage add value? | 3 × ~40 min |
| **A2: Factor type** | D_L only, D_A only, D_I only — which factor matters most? | 3 × ~40 min |
| **A2b: `min_overlap_frames` 4 vs 8** | 4/16 (2659 tubes, noisier) vs 8/16 (~1000-1500, cleaner) — switch if +Prec@K | 2 × ~40 min |
| **A3: Surgery vs naive fine-tune** | Same layers unfrozen, raw clips (no factors) — is factoring the key? | ~40 min |
| **A4: Random seeds** | 3-5 seeds of best config — statistical significance | 3 × ~40 min |
| **A5: D_I tube crop type** | Centroid-30%-square vs tight-union-bbox (m10 `per_object_bboxes_json`) — does identity-aware cropping help? | 2 × ~40 min |

Priority if time-constrained: **A3** (proves factoring matters) then **A4** (NeurIPS rigor).

---

## 📋 Backlog

- ⬜ 🟡 Paper figures: per-clip segmentation samples (m08 CPU-only)
- ⬜ 🟡 Verification videos: MP4 with mask overlay for temporal consistency
- ⬜ 🟡 Output dir restructure: verify all cross-references after per-module migration
- ⬜ 🟡 **D_I gold-standard architecture** (post-deadline): current tight-union-bbox crop is still a POC shortcut. Gold standard (Social-Fabric ICCV'21, Video-HOI NeurIPS'22): per-agent tubelets + RoIAlign on scene features + pair transformer. Requires V-JEPA forward-pass change → out of NeurIPS scope, log for v2.
- ⬜ 🟡 **m11 GPU rewrite** (post-POC): replace `scipy.ndimage.gaussian_filter` (σ=15 blur, ~2.4s/clip CPU bottleneck) with `kornia.filters.gaussian_blur2d` on GPU — projects 3.5s/clip → ~5ms/clip compute, FULL 115K from 5.6d → ~10h. Needs kornia in requirements_gpu, ≥40dB PSNR diff-test vs scipy, `--gpu` CPU-optional flag.
- ⬜ 🟢 `hf_outputs.py` upload: `git_push.sh` doesn't `source .env`
- ⬜ 🟢 `setup_env_uv.sh`: cuML/SAM3 version ping-pong
- ⬜ 🟢 FA3 installation: only if SAM3 bottleneck on FULL
- ⬜ 🟢 `output_guard.py` absolute-path → repo-relative (errors_N_fixes.md #23): stops noisy HF 404 + URL-encoded `%2Fworkspace%2F...` log spam on every m10/m11 run.

---

## 🔧 Troubleshooting

| Problem | Fix | Time |
|---|---|---|
| Grounded-SAM box quality poor | Try YOLO-World + SAM instead | 2h |
| V-JEPA 2.1 shape mismatch | `state_dict.keys()` vs model params | 1h |
| LoRA target modules wrong | `print(model)` → find attn module names | 30 min |
| Surgery loss NaN | Lower LR, check grad norms | 1h |
| D_I: 0% clips have tubes | Lower `max_distance_frame_fraction` in YAML | 15 min |
| D_I: 100% clips have tubes | Raise `min_overlap_frames`, lower `tube_margin_pct` | 15 min |

---

## 🔮 Future (post-paper)

- ✅ Split m09 (2164 lines) → m09a_pretrain.py + m09b_explora.py + m09c_surgery.py + utils/training.py (2026-04-15, #49)
- WebDataset TARs for factor datasets (.npy won't scale to 115K)
- 6 interaction perturbations (tube jitter, margin random, raw/masked mixing)
- Patch shortcut sanity check (eval raw vs patched clips)
- Cooldown (64f) implementation

---

## ✅ Completed

### 2026-04-15 (~2h GPU): Path B speedup + bbox-tubes + m10 consolidation
- 5 bugs found & fixed: #37 DINO fp16 text-branch crash (fp32 default), #38 Sam3Tracker box depth=3 (not 4), #39 session.reset_tracking_data (not processor), #40 silent bug — object_score_logits not iou_scores, #41 add_text_prompt kwarg `text=` not `prompts=`
- transformers 4.57.6 → **5.5.4** (setup_env_uv.sh steps [9/10] DINO + [10/10] facebook/sam3 ~12 GB HF_TRANSFER parallel)
- HF `Sam3TrackerVideoModel` integrated; `max_frame_num_to_track` now works (raw sam3 pkg #33/#35 unfixable)
- m10 v2_HF merged back into `m10_sam_segment.py` (P-3a probe stripped); `train_surgery.sh` unchanged
- m11 D_I upgrade: `per_object_bboxes_json` saved by m10; `make_interaction_tubes_from_bboxes` replaces fixed 30% centroid square
- setup_env_uv.sh: added non-fatal `uv pip check` with allowlist (sam3/numpy, sam3/ftfy, torch/cuda-bindings, decord)
- preflight skill extended B16-B20 for transformers 5.x regression guards
- Measured on dense100: **11.02 s/clip (4.21× faster), 6146 agents (+10 %), 8723 D_I tubes (+228 %), 91 % clips have tubes**
- 115K FULL ETA: 61 days → **14.7 days on 24GB**, **3.7 days on 96GB+batch×4**

### 2026-04-14 (~5h GPU): Grounded-SAM Pivot + Level 2 multi-anchor
- 32 bugs found and fixed (see `errors_N_fixes.md` #18-32)
- Architecture pivot: SAM3-text-only → Grounded-SAM Path D (DINO + SAM3.1 text+boxes hybrid)
- **Level 2 upgrade**: single-anchor → multi-anchor DINO re-seed (4 anchors, drift capped at ≤2 frames)
- Fixed 17-cat agent taxonomy in `ch11_surgery.yaml` replacing per-clip VLM `notable_objects`
- DINO weights pre-cached via `setup_env_uv.sh` step [9/9] (~1.8 GB at HF_HOME)
- Density-scored 100-clip subset: `data/sanity_100_dense.json` (74 tier1 + 25 tier2)
- Top-20 2x2 MP4 video grid added to m11 for human eyeballing + website
- Verified D_L/D_A/D_I quality on 20 SANITY clips (39 D_I tubes from 9 clips)
- Tuned thresholds Option C for new mask distribution (recall-first)
- `verify_or_skip` completeness check fixed — partial runs now resume properly

### 2026-04-12/13 (~6h GPU): Initial SANITY infrastructure
- 17 initial bugs fixed (env, SAM3 integration, torchcodec SIGSEGV)
- Per-module output dirs: `outputs/{mode}/{module_name}/` for all m04-m11
- m10 overlay verification images + m11 2x2 per-clip grids implemented
- Composite quality gate (4 checks: pixel ratio, mask confidence, clips with agents)
- `--plot` flag on m10 and m11 for CPU-only plot regeneration

---

## ⏱️ Time Budget

| Phase | Hours | Status |
|---|---|---|
| Phase 0: Grounded-SAM pivot (done on GPU) | ~4h spent | ✅ |
| Phase 1: GPU SANITY (24GB → 96GB migration 2026-04-17) | ~11h spent | ✅ all steps done, SANITY chain fully green |
| Phase 2a: POC 100-dense (discovery tier — RETIRED) | ~4h spent | ✅ caught 2 config bugs (#60 max_epochs, #61 warmup). Student too overfit to publish. |
| Phase 2b: POC 1K val_1k (publishable tier) | ~10h projected | 🎯 NEXT — full pipeline D.2→E.3 with both config fixes in place |
| Decision gate | — | ⬜ |
| Phase 3: Scale 115K + ablations | 12h | ⬜ |
| Phase 4: Paper writing | 14h | ⬜ |
| Buffer | 2h | |

---

## 📁 Key Files

| File | What |
|---|---|
| `src/CLAUDE.md` | Codebase rules |
| `iter/iter8/runbook.md` | GPU execution commands |
| `iter/iter8/plan_training.md` | System design, architecture, literature |
| `iter/iter8/errors_N_fixes.md` | 17 bugs catalogued |
| `configs/train/ch11_surgery.yaml` | Surgery config (stages, thresholds) |
| `configs/model/vjepa2_1.yaml` | V-JEPA 2.1 model config |
| `iter/utils/literarure_survey.md` | 24 JEPA variants (fallback) |
