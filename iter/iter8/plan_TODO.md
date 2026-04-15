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

## 🔥 Active (Phase 1: GPU SANITY) — Steps C/D/E next

- ✅ Step A (SANITY 20-clip): m10 Grounded-SAM segmentation — quality gate PASS
- ✅ Step B (SANITY 20-clip): m11 factor datasets — D_L/D_A/D_I verified
- ✅ Step A' (POC 100 dense, v2_HF): 6146 agents, 8723 interactions, 11.02 s/clip
- ✅ Step B' (POC 100 dense, bbox-tubes): 91/100 D_I clips, 8723 tubes, 5659 unique shapes
- 🔥 Step C: m05 frozen V-JEPA 2.1 embedding on 100-clip dense subset (`--POC --subset data/sanity_100_dense.json`)
- 🔥 Step D: m09 ExPLoRA training on 100-clip dense subset
- 🔥 Step E: m09 Surgery training on 100-clip dense subset (uses `--factor-dir outputs/poc/m11_factor_datasets/`)
- ⬜ Commit all fixes via `git_push.sh`

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
- ⬜ 🟢 `hf_outputs.py` upload: `git_push.sh` doesn't `source .env`
- ⬜ 🟢 `setup_env_uv.sh`: cuML/SAM3 version ping-pong
- ⬜ 🟢 FA3 installation: only if SAM3 bottleneck on FULL

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

- Split m09 (~2000 lines) into m09_pretrain.py + m09b_explora.py + m09c_surgery.py
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
| Phase 1: GPU SANITY (24GB, $0.20/hr) | ~10h spent + ~1h remaining | 🔄 A/B/A'/B' done, C/D/E pending |
| Phase 2: GPU POC (96GB, $0.80/hr) | 3h | ⬜ |
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
