# Training Plan: Ch11 (Surgery Fine-Tuning) + Ch10 (Ablation Comparison)
> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**
- 🧬 Factor isolation curriculum (plan_training.md:275): D_L → D_A → D_I = "teach anatomy, then physiology, then surgery" — vs vanilla's "everything at once" which   entangles layout and agent gradients, diluting the learning signal on any single axis.                                                                                   
- 🛡️  Catastrophic-forgetting resistance by construction (PDF proposal Sec 11.5): progressive prefix unfreezing keeps deep layers frozen in Stages 1-2, so pre-trained ImageNet/WebVid features survive; vanilla full fine-tuning loses >70% of pre-training accuracy in 10 iterations on distribution shift (DINO ViT-B case, Sparse-Tuning    paper).
- 🎯 Selective capacity allocation with replay (Ch11 proposal): stage mixture (90% D_A + 10% D_L replay → 85% D_I + 10% D_A + 5% D_L) = controlled interpolation between old + new objectives; vanilla has no replay so loses old factor signal the moment new data dominates.                                                                    
  
> Ch11 (surgery on frozen) is the PRIMARY path. Ch10 (brute-force) is a paper comparison arm, run LATER.
> Ref: `Literature/proposal/FactorJEPA/FactorJEPA.md` Sections 10-11
> **If surgery doesn't improve metrics:** See `iter/utils/literarure_survey.md` — 24 JEPA variants surveyed. Top fallback techniques: SIGReg regularizer (LeJEPA, replaces EMA), leakage-free factor training (VLA-JEPA), temporal straightening diagnostic (LeWorldModel).

## 🟢 Status (2026-04-26): iter11 v2 LR re-anchored to 5e-5, 15-epoch budget, runs in flight

- 🔬 **2026-04-26 LR re-anchoring (`errors_N_fixes.md #78`)**: after `surgery_2stage_noDI` v2 (5 epochs @ lr=1e-5) ended with val_jepa still descending (0.4663 final, curve un-plateaued) and probe Prec@K saturated (Δ +0.22 over 27 probes ≪ ±3.78 BCa CI), re-anchored `base_optimization.yaml` to Meta's continual training recipe BS-scaled (4.25e-4 @ BS=256 → **5.0e-5 @ BS=32**). Five inherited yaml changes: `lr: 1e-5 → 5e-5`, `max_epochs.full: 5 → 15`, `warmup_cap_pct: 10 → 15`, `grad_clip: 10.0 → 1.0`, `nan_tolerance: 3 → 2`. Dead `lr_schedule: constant` field deleted (build_scheduler is unconditionally cosine-to-`min_lr=1e-6`). SANITY (1-epoch FULL-mode at 5e-5) verified: val_jepa 0.4835 → 0.4693 (Δ−0.014 in 1 ep ≈ matches 1e-5 final after 5 ep), Prec@K 75.05–75.22, 0 NaN, max post-warmup grad=0.803. Two 15-epoch FULL runs in flight on 2 boxes (`surgery_2stage_noDI_epoch15_v3` + `surgery_3stage_DI_epoch15_v2`); explora + surgery_2stage_loud_agent queued.

## 🟢 Status (2026-04-25): iter11 v2 infra landed, ready to launch

- ✅ **iter10 closed 2026-04-23**: v10/v13/v14/v15a/v15b all Δ ≈ 0 pp on eval_10k (N=9,297, CI_half ±0.42 pp, paired BCa p ≥ 0.68). See `iter/utils/experiment_log.md` cross-run table. No variant cleared the +3 pp gate. v15c WITHDRAWN (silent L/A renorm, `errors_N_fixes.md #73`); ckpt deleted.
- 🗑️ **iter11 v1 INVALIDATED 2026-04-24**: m09b ExPLoRA first 10K run halted at step 174/298 by `prec_plateau_enabled` trigger whose threshold (0.3 pp) sat below the CI_half noise floor (1.73 pp at N=1000) → fired on sampling jitter, not stagnation. Ckpts + eval dirs deleted.
- 🚀 **iter11 v2 infra LANDED 2026-04-25** — ready to launch: 4-variant apples-to-apples at 10K (ExPLoRA + 3 surgery recipes) with unified 5-epoch budget, `saves_per_epoch=5`, val-loss plateau as the ONLY active early-stop (per `feedback_only_val_loss_early_stop.md` — probe-Prec@K triggers all operate below BCa CI_half noise). Each train yaml carries a `data:` block (module / model_config / subsets / local_data / output_dir / eval paths / adapted_encoder) so 3 thin wrappers — `scripts/run_factor_prep.sh` + `run_train.sh` + `run_eval.sh` (+ `lib/yaml_extract.py`) — take ONLY yaml paths as args (CLAUDE.md "No hardcoded paths" rule). Terminal commands: `iter/iter11/runbook.md`. Code-dev plan: `iter/iter11/plan_code_dev.md`.
- 🔮 **FUTURE — Plan B "true-hard" re-curation (deferred 2026-04-26, post-iter11-v3)**: mid-training diagnostic on `ultra_hard_3066_val` showed probe Prec@K saturated at ~75 vs ~28-30 on iter10's random subset_10k. The `ultra_hard` curation rule (≥4 Hard triggers AND ≥4 Indian-specific objects) selected for **tag-richness**, which is the opposite of retrieval-difficulty — tag-rich clips trivially retrieve same-tag neighbors via kNN, leaving no headroom (Δ ceiling ≤ 5 pp vs CI_half ±2.4 pp at N=306, never significant). **Plan B**: build `src/m00g_frozen_hard.py` to encode a 50K candidate pool with frozen V-JEPA 2.1, compute per-clip Prec@K, select bottom quartile (`prec_at_k < 0.30`) as the genuinely-retrieval-hard subset. Then 80/10/10 split + re-run the whole pipeline. Cost: ~3 h GPU re-curation + ~10-15 h GPU re-train. **Trigger**: execute ONLY AFTER current iter11 v3 `run_eval.sh` completes for the 3,066 selected clips (decision rule + step list in `iter/iter11/plan_TODO.md`). Not a current-session change.
- 🎯 **iter11 v3 hard-pivot — splits ready 2026-04-25**: per-clip post-hoc Δ analysis showed Surgery's signal concentrates on the Indian agent-rich tail. `src/m00f_category_subsets.py` built 9 category subsets + a 3-way 80/10/10 train/val/eval split inside `data/ultra_hard_3066.json` (3,066 clips meeting ≥4 Hard triggers AND ≥4 Indian-specific objects). Two download strategies via `src/m00d_download_subset.py` — verified against HF dataset (130 GB / 115,687 clips = **1.13 MB/clip**): **Strategy A** = 3,066-clip parent only (~4 GB, ~7.5 min, measured); **Strategy B** = all 9 categories (~99K unique, ~**109 GB** — corrected from earlier 250 GB hallucination, ~70 min). Both run on CPU+network, 0 h GPU. Hand-off: each train yaml's `data:` block points `*_local_data` at `data/ultra_hard_3066_local/` and `*_subset` at the 3 split JSONs → existing `run_factor_prep.sh → run_train.sh → run_eval.sh` chain consumes unchanged.
- ✅ **Prior validations** carried forward into iter11 v2: H2 stratified splits (data/val_500.json + data/test_500.json files retained but iter11 v2 probes against the FULL `data/val_1k.json` instead — paired-comparison fairness); per-stage plateau reset (iter10 #70); typed-interactions obj_id→cat persistence (#77); fingerprinted m05 surgical-ckpt paths (#75); fail-loud per-factor preflight in `StreamingFactorDataset` (#73).
- 🔒 **50K scale-up escalation**: conditional on iter11 v2 best Δ ∈ [+0.3, +3) pp on eval_10k. Builds `data/subset_50k.json` disjoint from 10K+val+eval_10k. Budget ~52 h / ~$42.
- 🔒 **Concede tier** (if iter11 v2 all Δ < +0.3 pp): re-pitch paper as narrower **"layout-factor surgery at NOISE FLOOR"** — D_L+D_A+D_I move from recipe stages to ablations table; pretrained-feature preservation (BWT ≈ 0 across scales) becomes the headline.
- ✅ **Streaming factor generation landed 2026-04-19** (9 files, ~1014 LoC, 10/10 bitwise parity on iter8 D_L/D_A .npy): `StreamingFactorDataset(IterableDataset)` generates D_L/D_A on-demand from `(raw_mp4, mask.npz)` pairs inside m09c's `DataLoader(num_workers=16, persistent_workers, prefetch_factor=4)`. m11 `--streaming` flag short-circuits non-verify clips → ~90 % m11 wall reduction. **Unlocks full 10K→50K→115K ladder on a single 500 GB instance** (was 500 GB → 3 TB → 5 TB tier progression). Projected 10K wall @ num_workers=16: **4.77 h** (Tier-3 regression test). See `iter/iter9/plan_code_dev.md`.
- ✅ **iter11 v2 val/eval split policy**: probe (mid-training) on `data/val_1k.json` (full 1K, paired-comparison fair across all 4 variants); decision gate on `data/eval_10k.json` (paired BCa, N=9,297, CI_half ±0.42 pp). val_500/test_500 files retained for iter9 reproducibility but no longer used by iter11 v2. iter9 H2 best-of-K bias mitigation context: `val_1k.json` → `val_500.json` (probe) + `test_500.json` (gate, touched once); CI ±1.5 pp → ±2.1 pp at N=500. Iter11 v2 supersedes by tightening to N=9,297 paired BCa (~5× tighter CI).
- ✅ **Disk cleanup freed ~123 GB pre-launch**: iter8 1K POC outputs (67 GB) archived to HF + deleted; iter8 archive's D_L/D_A/D_I bulk .npy (56 GB) permanently retired by streaming refactor; git gc compacted packs.


- ✅ **Full 1K POC pipeline completed** (2026-04-19, A→F on val_1k): m10 1000/1000 in 1h00m, m11 factor-gen in 11m43s, m09c Surgery v3 (3-stage, 5 ep) in 2h27m, m05 frozen+surgical on 100-val hold-out in ~7 min combined, m06 decision gate in ~1 min.
- ❌ **Decision gate FAILED at N=100**: Frozen Prec@K = 20.17 ±4.5 pp vs Surgical = 20.33 ±4.67 pp → Δ +0.17 pp, **CIs overlap heavily**. N=100 half-width ±4.5 pp is too wide to resolve any sub-5-pp Surgery-vs-Frozen delta. See `iter/iter8/status_1k_poc_run.md` for full plot set + diagnosis.
- 🩺 **Stage 3 diagnosis (multi-signal)**: best Prec@K = 20.50 @ step 12 (Stage 1), held through Stage 2, **dropped to 19.83–20.00 in Stage 3** (BWT=−0.33). Val_jepa Δ/step: Stage 1 −0.5 %, Stage 2 −1.4 %, **Stage 3 −0.3 %** (4× slower optimization per step). Cycle@K drifts 63 → 62. Zero new best-ckpt events in 27 Stage-3 probes. Stage 3 is net-useless at this scale. See `iter/utils/experiment_log.md` 2026-04-19 entry.
- ✅ **Surgery recipe revised 2026-04-19**: **2 stages** (Stage 3 dropped), Stage 2 replay bumped 10 % → 30 % D_L (closer to CLEAR 50/50 recipe), `max_epochs: 1` across all modes, `batch_size: 32` research-LOCKED (published V-JEPA 2.1 Adam co-tuning preserved). D_I factor still generated by m11 for later ablation.
- ✅ **Early-stop suite landed**: (1) catastrophic-forgetting kill-switch (5 pp × 3 probes, existing), (2) **plateau** on val_jepa (Δ < 1e-3 × 5 probes), (3) **negative-BWT** trigger (BWT < −0.5 pp × 10 probes). Unified `kill_state.reason` in `training_summary.json.early_stop`.
- ✅ **`use_permanent_val` flag** added: FULL/10K+ modes train on the full training set (no internal 90/10 split) and validate on permanent `data/val_1k_local` (N=1000, CI ~±1.5 pp — 3× tighter than POC's ±4.5 pp). Fail-loud on any train/val clip overlap. POC 1K still uses internal split (trains on val_1k itself as smoke test).
- ✅ **Plot fixes**: `m09_val_loss.png` no longer clobbered by empty `plot_training_curves` overwrite (utils/plots.py skips empty val source). `m09_train_loss.png` gains dual x-axis (bottom = optimizer step, top = `n_unique × n_epochs` training-samples-seen). `_render_live_plots` refreshes 3 val plots every probe (silent).
- 📈 **Plateau-seeking scale-ladder**: 10K → 50K → 115K is a gated ladder, not a commitment to full 115K. Each tier × eval_10k Prec@K = one point on a publishable **Prec@K vs log₁₀(n_clips)** curve with 95 % CIs. Val-loss plateau auto-halts GPU when signal saturates. Gate per tier: Δ ≥ 0.5 pp vs prior tier → scale; else that tier is the publishable knee.
- 📅 **Deadline fit (NeurIPS May 04, 10 days remaining from 2026-04-24)**: iter11 v2 (~30-60 h train + ~10 h eval) lands scale 1. 50K (~52 h) only if iter11 v2 gate passes (Δ ≥ +0.3 pp on eval_10k). 115K only if 50K ≥ +0.5 pp vs 50K. If iter11 v2 all Δ < +0.3 pp, 10K becomes the "concede-tier" submission — no 50K/115K spend.
- 🎯 **Decision gate (iter11 v2)**: eval_10k paired BCa Δ Prec@K with non-overlapping 95 % CIs → consult escalation in `plan_TODO.md`; else concede-tier per `iter/utils/literarure_survey.md` fallbacks.

---

## System Design: Full Pipeline (m00 → m11 → eval)

```mermaid
flowchart TB
    subgraph DATA ["Data Pipeline (m00-m03, CPU, done)"]
        direction LR
        YT["m00: YT videos<br>714 videos"] --> DL_V["m01: download<br>480p"]
        DL_V --> SCENE["m02: scene split<br>4-10s clips"]
        SCENE --> PACK["m03: WebDataset<br>116 TARs → HF"]
    end

    subgraph EVAL_CH9 ["Ch9 Eval Pipeline (m04-m08b, done)"]
        direction LR
        TAG["m04: VLM tag<br>Qwen3-VL<br>→ tags.json"]
        MOT["m04d: RAFT<br>motion features"]
        EMB["m05: V-JEPA 2.1<br>frozen embed<br>→ (N, 1664)"]
        BASE["m05b: baselines<br>DINOv2, CLIP,<br>shuffled"]
        FAISS["m06: FAISS-GPU<br>9 metrics<br>Prec@K, nDCG@K"]
        TEMP["m06b: temporal<br>correlation"]
        UMAP["m07: cuML UMAP"]
        PLOT["m08: plots"]
        COMP["m08b: compare<br>radar + table"]
    end

    subgraph CONFIGS ["configs/"]
        direction TB
        MODEL["model/<br>vjepa2_1.yaml<br>(2B, 1664-dim)"]
        TRAIN_E["train/<br>explora.yaml"]
        TRAIN_S["train/<br>surgery_2stage_noDI.yaml<br>(+ loud_agent / 3stage_DI)"]
        PIPE["pipeline.yaml<br>encoders, limits"]
    end

    subgraph EXPLORA ["ExPLoRA Training (Step 1b, m09)"]
        direction TB
        E_CLIP["Raw Indian clip<br>16 frames, 384x384"]
        E_MASK["Mask 80%<br>8+2 blocks"]
        E_STU["Student ViT-G 2B<br>blocks 0-1: TRAINABLE<br>blocks 2-47: FROZEN + LoRA<br>~5% params trainable"]
        E_TEA["Teacher (EMA)<br>no grad"]
        E_LOSS["JEPA L1 loss<br>predict masked tokens"]
        E_OUT["student_encoder.pt<br>vjepa_2_1_explora"]
        E_CLIP --> E_MASK --> E_STU --> E_LOSS
        E_TEA --> E_LOSS
        E_LOSS --> E_OUT
    end

    subgraph SURGERY ["Surgery Training (Step 2, m10 → m11 → m09)"]
        direction TB

        S_SAM["m10: SAM 3.1<br>text prompt per clip<br>from tags.json<br>notable_objects"]
        S_DL["m11: D_L<br>layout-only<br>blur agents"]
        S_DA["m11: D_A<br>agent-only<br>suppress BG"]
        S_DI["m11: D_I<br>interaction tubes<br>agent pairs within d_max"]

        S1["m09 Stage 1<br>layers 0→12 trainable<br>100% D_L"]
        S2["m09 Stage 2<br>layers 0→24 trainable<br>90% D_A + 10% D_L"]
        S3["m09 Stage 3<br>layers 0→36 trainable<br>85% D_I + 10% D_A + 5% D_L"]
        S_OUT["student_encoder.pt<br>vjepa_2_1_surgical"]

        S_SAM --> S_DL
        S_SAM --> S_DA
        S_SAM --> S_DI
        S_DL --> S1 --> S2
        S_DA --> S2 --> S3
        S_DI --> S3 --> S_OUT
    end

    subgraph RE_EVAL ["Re-Evaluation (all encoders)"]
        direction LR
        RE_EMB["m05: re-embed<br>frozen / ExPLoRA / surgical"]
        RE_M6["m06: Prec@K<br>with 95% CI"]
        RE_COMP["m08b: compare<br>frozen vs ExPLoRA<br>vs surgical"]
        RE_EMB --> RE_M6 --> RE_COMP
    end

    subgraph OUTPUTS ["outputs/poc/"]
        direction TB
        O_FROZEN["embeddings_vjepa_2_1_frozen.npy"]
        O_EXPLORA["embeddings_vjepa_2_1_explora.npy"]
        O_SURGICAL["embeddings_vjepa_2_1_surgical.npy"]
        O_METRICS["m06_metrics_*.json"]
        O_FACTORS["factors/<br>masks/ D_L/ D_A/ D_I/"]
    end

    DATA --> EVAL_CH9
    DATA --> EXPLORA
    DATA --> SURGERY
    TAG --> S_SAM
    CONFIGS --> EXPLORA
    CONFIGS --> SURGERY
    E_OUT --> RE_EVAL
    S_OUT --> RE_EVAL
    RE_EVAL --> OUTPUTS

    style YT fill:#5e35b1,color:#fff,font-weight:bold
    style DL_V fill:#5e35b1,color:#fff,font-weight:bold
    style SCENE fill:#5e35b1,color:#fff,font-weight:bold
    style PACK fill:#5e35b1,color:#fff,font-weight:bold
    style TAG fill:#00acc1,color:#fff,font-weight:bold
    style EMB fill:#43a047,color:#fff,font-weight:bold
    style FAISS fill:#e53935,color:#fff,font-weight:bold
    style COMP fill:#b71c1c,color:#fff,font-weight:bold
    style E_STU fill:#1565c0,color:#fff,font-weight:bold
    style E_LOSS fill:#1565c0,color:#fff,font-weight:bold
    style E_OUT fill:#1565c0,color:#fff,font-weight:bold
    style S_SAM fill:#c62828,color:#fff,font-weight:bold
    style S_DL fill:#1b5e20,color:#fff,font-weight:bold
    style S_DA fill:#e65100,color:#fff,font-weight:bold
    style S_DI fill:#6a1b9a,color:#fff,font-weight:bold
    style S1 fill:#1b5e20,color:#fff,font-weight:bold
    style S2 fill:#e65100,color:#fff,font-weight:bold
    style S3 fill:#6a1b9a,color:#fff,font-weight:bold
    style S_OUT fill:#c62828,color:#fff,font-weight:bold
    style MODEL fill:#616161,color:#fff,font-weight:bold
    style TRAIN_E fill:#1565c0,color:#fff,font-weight:bold
    style TRAIN_S fill:#c62828,color:#fff,font-weight:bold
    style O_FROZEN fill:#546e7a,color:#fff,font-weight:bold
    style O_EXPLORA fill:#1565c0,color:#fff,font-weight:bold
    style O_SURGICAL fill:#c62828,color:#fff,font-weight:bold
```

**Layman story (full pipeline):** Imagine building a map of every street in India for a self-driving car that only knows American roads.

1. **Data (purple, top):** You record 714 walking tour videos across Indian cities, chop them into 115K short clips (10 seconds each), and upload to a cloud dataset.

2. **Eval (teal/green/red, left):** A robot VLM watches each clip and tags it: "market, day, crowded, auto-rickshaw, sacred cow." Then V-JEPA 2.1 (the American-trained brain) converts each clip into a 1664-number fingerprint. FAISS finds which clips have similar fingerprints. If "market" clips cluster together → the brain understands markets. If not → it's confused.

3. **ExPLoRA (blue, middle):** Bolt tiny adapter modules (LoRA) onto the frozen brain. Only 5% of parameters change. Show it Indian clips, same fill-in-the-blanks game as V-JEPA's original training. Quick and cheap (~1 hour). This is the BASELINE TO BEAT.

4. **Surgery (red, middle-right):** THE EXPERIMENT.
   - **m10 (SAM 3.1):** An AI eye surgeon (SAM 3.1) looks at each clip's tags ("auto_rickshaw, pedestrian") and cuts the video into: roads-only (D_L), people-only (D_A), and interactions (D_I — a pedestrian crossing in front of an auto-rickshaw).
   - **m11:** Generates the 3 patched versions of each clip.
   - **m09 (3 stages):** Teaches the brain in order: first roads (layers 0-12), then people (layers 0-24), then interactions (layers 0-36). Each stage unlocks deeper layers. Earlier concepts are replayed to prevent forgetting.

5. **Re-eval (bottom):** Re-run the fingerprint + FAISS test on all 3 brains (frozen, ExPLoRA, surgical). The winner: whichever brain makes the best "market" clusters.

---

## System Design: ExPLoRA (Step 1b)

```mermaid
flowchart LR
    subgraph DATA ["1. Data"]
        CLIP["Indian clip<br>10s, 16 frames<br>from WebDataset"]
        CLIP --> AUG["Augment<br>RRC 384x384<br>h-flip"]
    end

    subgraph MASK ["2. Mask"]
        AUG --> TOK["Patchify<br>24x24x8<br>=4608 tokens"]
        TOK --> VIS["visible<br>~900 tokens"]
        TOK --> HID["masked<br>~3700 tokens"]
    end

    subgraph MODEL ["3. ExPLoRA Model"]
        VIS --> STU["Student ViT-G 2B<br>blocks 0-1: TRAINABLE<br>blocks 2-47: FROZEN + LoRA<br>rank=16, ~5% params"]
        AUG2["same clip"] --> TEA["Teacher ViT-G 2B<br>ALL tokens<br>no grad (EMA copy)"]
        STU --> PRED["Predictor<br>24-layer 384-dim"]
    end

    subgraph LOSS ["4. Loss + Update"]
        PRED --> L1["L1 loss<br>pred vs teacher<br>at masked positions"]
        TEA --> L1
        L1 --> ADAM["AdamW<br>blocks 0-1 + LoRA only"]
        ADAM --> EMA["EMA update teacher<br>tau=0.99925"]
    end

    style CLIP fill:#5e35b1,color:#fff,font-weight:bold
    style AUG fill:#00897b,color:#fff,font-weight:bold
    style VIS fill:#2e7d32,color:#fff,font-weight:bold
    style HID fill:#c62828,color:#fff,font-weight:bold
    style STU fill:#1565c0,color:#fff,font-weight:bold
    style TEA fill:#546e7a,color:#fff,font-weight:bold
    style PRED fill:#6a1b9a,color:#fff,font-weight:bold
    style L1 fill:#c62828,color:#fff,font-weight:bold
    style ADAM fill:#1565c0,color:#fff,font-weight:bold
    style EMA fill:#9c27b0,color:#fff,font-weight:bold
```

**Layman story (ExPLoRA):** A fill-in-the-blanks exam on Indian streets. The brain is FROZEN except for 2 "input processing" blocks and tiny LoRA adapters (~5% of total parameters). The student sees 20% of each video and guesses the hidden 80%. The teacher holds the answer key. Only the adapters and 2 blocks learn — like learning a new accent without forgetting the language. Cheap (~20 min on 1K clips), proven (+8% on satellite imagery).

---

## System Design: Surgery (Step 2) — THE PAPER NOVELTY

> **2026-04-14 update:** m10 architecture pivoted to **Grounded-SAM (Path D)** — Grounding DINO open-vocab box detection on frame 0 + SAM 3.1 text-tracked + box-refined propagation across 16 frames. Replaces the original SAM 3.1 native text grounding which failed on Indian objects (10/15 clips wrong/missing masks). Fixed 17-category agent taxonomy in `configs/train/surgery_*.yaml > factor_datasets.grounding_dino.agent_taxonomy` (was `ch11_surgery.yaml` pre-iter11-v2 rename) replaces per-clip VLM `notable_objects`. See `errors_N_fixes.md` #20-27 for pivot history.

```mermaid
flowchart TB
    subgraph SAM ["m10: Grounded-SAM Segmentation (GPU)"]
        CLIP["Indian clip<br>16 frames"] --> DINO["Grounding DINO Base<br>17-cat compound prompt<br>'pedestrian. car. bus...'<br>frame 0 only"]
        DINO --> BOXES["Boxes per category<br>+ confidence scores"]
        BOXES --> SAM31["SAM 3.1 add_prompt<br>text=cat (tracking)<br>+ boxes_xywh_norm (refine)<br>+ box_labels=[1]*N"]
        SAM31 --> AMASK["Agent masks<br>per-frame<br>(propagated across 16f)"]
        AMASK --> LMASK["Layout masks<br>= NOT agent_mask"]
    end

    subgraph FACTOR ["m11: Factor Datasets (CPU)"]
        CLIP2["Original frames"] --> DL["D_L: layout-only<br>blur agents<br>(Gaussian sigma=15)"]
        CLIP2 --> DA["D_A: agent-only<br>suppress background<br>(soft matte x0.1)"]
        AMASK --> DL
        LMASK --> DA
        AMASK --> MINE["Interaction mining<br>agent pairs within<br>20% frame width<br>for >= 4 frames"]
        MINE --> DI["D_I: interaction tubes<br>cropped to bounding box<br>of interacting agents"]
    end

    subgraph STAGE1 ["m09 Stage 1: Layout (layers 0-12)"]
        DL --> S1_IN["100% D_L clips"]
        S1_IN --> S1_STU["Student ViT-G 2B<br>layers 0-12: TRAINABLE<br>layers 13-47: FROZEN"]
        S1_STU --> S1_L["JEPA L1 loss<br>same mask game"]
    end

    subgraph STAGE2 ["m09 Stage 2: Agents (layers 0-24)"]
        DA --> S2_A["90% D_A clips"]
        DL2["10% D_L replay"] --> S2_A
        S2_A --> S2_STU["Student ViT-G 2B<br>layers 0-24: TRAINABLE<br>layers 25-47: FROZEN"]
        S2_STU --> S2_L["JEPA L1 loss"]
    end

    subgraph STAGE3 ["m09 Stage 3: Interactions (layers 0-36)"]
        DI --> S3_I["85% D_I clips"]
        DA2["10% D_A replay"] --> S3_I
        DL3["5% D_L replay"] --> S3_I
        S3_I --> S3_STU["Student ViT-G 2B<br>layers 0-36: TRAINABLE<br>layers 37-47: FROZEN"]
        S3_STU --> S3_L["JEPA L1 loss"]
    end

    S1_L -->|"checkpoint"| STAGE2
    S2_L -->|"checkpoint"| STAGE3
    S3_L --> EXPORT["student_encoder.pt<br>vjepa_2_1_surgical"]

    style CLIP fill:#5e35b1,color:#fff,font-weight:bold
    style SAM31 fill:#c62828,color:#fff,font-weight:bold
    style AMASK fill:#e65100,color:#fff,font-weight:bold
    style LMASK fill:#1565c0,color:#fff,font-weight:bold
    style DL fill:#1b5e20,color:#fff,font-weight:bold
    style DA fill:#e65100,color:#fff,font-weight:bold
    style MINE fill:#6a1b9a,color:#fff,font-weight:bold
    style DI fill:#6a1b9a,color:#fff,font-weight:bold
    style S1_STU fill:#1b5e20,color:#fff,font-weight:bold
    style S1_L fill:#1b5e20,color:#fff,font-weight:bold
    style S2_STU fill:#e65100,color:#fff,font-weight:bold
    style S2_L fill:#e65100,color:#fff,font-weight:bold
    style S3_STU fill:#6a1b9a,color:#fff,font-weight:bold
    style S3_L fill:#6a1b9a,color:#fff,font-weight:bold
    style EXPORT fill:#c62828,color:#fff,font-weight:bold
```

**Layman story (Surgery):** Teaching a foreign doctor to work in an Indian hospital — in 3 stages:

**Stage 1 — Layout (green):** Show the doctor ONLY the hospital — walls, floor, equipment, wiring. No patients, no staff. Blur out all people. The doctor's "early visual processing" (layers 0-12) learns Indian infrastructure — narrow lanes, overhead wires, speed breakers, open drains. The rest of the brain stays FROZEN. Training data: D_L (layout-only clips where agents are blurred).

**Stage 2 — Agents (orange):** Now show the patients and staff — people, vehicles, animals. The background is dimmed to 10% brightness. The doctor's "mid-level understanding" (layers 0-24) learns to recognize Indian agents — auto-rickshaws, sacred cows, street vendors, cycle rickshaws. Mix in 10% layout-only clips so the doctor doesn't forget the infrastructure learned in Stage 1.

**Stage 3 — Interactions (purple):** Finally, show HOW people interact with each other and with the environment — a pedestrian dodging an auto-rickshaw, a vendor blocking half the road, a cow calmly walking through a busy market. These are interaction tubes: cropped video segments showing exactly where two agents are close together for at least 4 frames. The doctor's "high-level reasoning" (layers 0-36) learns Indian interaction patterns. Mix in 10% agent + 5% layout clips for replay.

**Why 3 factors and not just raw clips?** Each factor ISOLATES one concept. D_L teaches infrastructure WITHOUT confounding agent patterns. D_A teaches agents WITHOUT confounding layout. D_I teaches interactions in CONTEXT. This is like teaching anatomy, then physiology, then surgery — not throwing everything at the student at once.

**Why progressive unfreezing?** Earlier ViT layers learn low-level features (edges, textures). Later layers learn high-level semantics (scene composition, interactions). By unfreezing more layers at each stage, the model learns simple Indian features first, then builds complex understanding on top. Replay mixing prevents catastrophic forgetting of earlier stages.

---

## V-JEPA Training: What's Actually Used

PPO/DPO/GRPO are RLHF methods for text-generating LLMs. They are **fundamentally inapplicable** to V-JEPA. V-JEPA is a deterministic encoder (video → embedding), not a generative model. There's no reward signal, no preference pairs, no policy to optimize.

### V-JEPA 2.0 vs 2.1 Training Components

| Component | V-JEPA 2.0 | V-JEPA 2.1 |
|-----------|-----------|-----------|
| **Loss** | L1 latent prediction (masked tokens only) | Dense Predictive Loss (ALL tokens, L1) |
| **Optimizer** | AdamW | AdamW |
| **LR Schedule** | Warmup-constant-cooldown (NOT cosine) | Same |
| **EMA** | Fixed momentum (no ramp-up) | Same |
| **Architecture** | Student-teacher with predictor | Same + deep self-supervision at intermediate layers |

Sources: [V-JEPA 2 (arXiv:2506.09985)](https://arxiv.org/abs/2506.09985), [V-JEPA 2.1 (arXiv:2603.14482)](https://arxiv.org/abs/2603.14482)

---

## Self-Supervised Video Encoder Training Algorithms

| Algorithm | Loss Type | Used By | Negatives? |
|-----------|-----------|---------|------------|
| **JEPA latent prediction (L1)** | Regression in latent space | V-JEPA 2/2.1 | No |
| DINO + iBOT | Cross-entropy (CLS + patch) | DINOv2 | No (EMA teacher) |
| MSE pixel reconstruction | Pixel regression | VideoMAE, MAE | No |
| BYOL | MSE normalized projection | BYOL | No (EMA) |
| InfoNCE / NT-Xent | Contrastive | SimCLR, MoCo | Yes |

---

## Continual Pretraining Approaches (Ch10)

Proposal (Sec 10.3) specifies: same JEPA loss on Indian clips, student-teacher EMA, optional drift control.

Standard approaches in literature:

| # | Approach | How it works | Relevance |
|---|----------|-------------|-----------|
| 1 | **Same SSL loss on new data** | Resume pretraining with JEPA loss on Indian clips | Most direct. V-JEPA 2 itself does stage-wise training (pretrain → post-train). **Our primary approach.** |
| 2 | **EWC (Elastic Weight Consolidation)** | Penalty on important weights from prior training | Prevents catastrophic forgetting. Our drift control (λ·‖θ-θ₀‖²) is equivalent to L2-anchored EWC. |
| 3 | **Knowledge distillation** | Frozen original model as teacher, adapted model matches teacher outputs + learns from new data | Confirmed for CLIP/DINOv2 continual learning. Could supplement JEPA loss. |
| 4 | **LoRA / adapters** | Freeze backbone, train low-rank adapter modules | Reduces trainable params. C-LoRA confirmed for continual vision learning. |
| 5 | **Frozen encoder + new predictor** | Freeze encoder, train only predictor on new data | V-JEPA 2's own action-conditioned post-training uses this. Cheapest option. |

---

## Surgery Fine-Tuning Approaches (Ch11)

Proposal (Sec 11.5) originally specified **3-stage** progressive prefix unfreezing (Layout → Agent → Interaction). **2026-04-19 1K POC revealed Stage 3 is net-useless at that scale** (BWT=−0.33, 4× slower val_jepa optimization, 0 new best-ckpt events, Cycle@K drifts down) → recipe revised to **2 stages** for 10K/50K/115K runs. D_I factor still generated by m11 for a single ablation run at FULL.

### Current (2-stage) recipe — active

| Stage | Layers Unfrozen | Input | Factor | Step-budget share |
|-------|----------------|-------|--------|-------|
| 1 | 0 to 25% of L | 100% D_L | Roads, buildings, wires | 50 % |
| 2 | 0 to 50% of L | 70% D_A + 30% D_L replay | Vehicles, people, animals | 50 % |

Why 30 % replay (was 10 %): CLEAR (NeurIPS 2018) recommends 50/50 novel-replay, reports insensitivity in [20-50 %]. Old 10 % sat below noise floor → Stage-1 layout signal was lost at the Stage 1→2 transition. See `configs/train/surgery_2stage_noDI.yaml` comment header for full rationale (was `ch11_surgery.yaml` pre-iter11-v2 rename).

### 🟡 v10 empirical finding: D_L dominance at current mask quality + LR

v10 (2026-04-20, 297 steps) produced **all 3 best-ckpt events in Stage 1 (D_L), zero in Stage 2 (D_A-dominated)**. This means that under current mask quality (concept_recall=0.655) AND the conservative 1e-6 LR, D_L is the only factor the model can productively train on — D_A's gradient signal is drowned by its own mask noise. Two v11 interventions target D_A's weakness directly:

- **H1** (LR 1e-6 → 1e-5): V-JEPA 2.1 paper default; 10× LR unlocks enough gradient magnitude for the model to learn D_A's harder-to-fit features in the face of mask noise.
- **H4** (DINO `box_threshold: 0.15 → 0.20`, `text_threshold: 0.12 → 0.18`): tighter detection reduces false-positive agent boxes → cleaner D_A matte → better D_A target signal. Requires fresh Step A (new m10 masks).

**Fallback framing — if H1+H4 still fail** (0 best-ckpt in Stage 2, BWT still negative): D_L dominance is structural, not a hyperparameter artifact. Land `surgery.stage1_only: true` opt-in flag, drop D_A from the recipe, and re-pitch the paper as **"layout-factor surgery"**:

| Stage | Layers Unfrozen | Input | Factor | Step-budget share |
|-------|----------------|-------|--------|-------|
| 1 | 0 to 25% of L | 100% D_L | Roads, buildings, wires | 100 % |

Paper claim shifts from "progressive layout→agent→interaction curriculum" to "**layout-only prefix-unfreezing preserves pretrained features while re-grounding the encoder on the target distribution's scene layout**". Narrower, but still novel — D_A becomes part of the **ablations** table (demonstrating that the bottleneck is mask quality, not the factor-decomposition idea) rather than the recipe.

### Original (3-stage) recipe — retired, but D_I kept for ablation

| Stage | Layers Unfrozen | Input | Factor |
|-------|----------------|-------|--------|
| 1 | 0 to 25% of L | 100% D_L | Roads, buildings, wires |
| 2 | 0 to 50% of L | 90% D_A + 10% D_L | Vehicles, people, animals |
| 3 | 0 to 75% of L | 85% D_I + 10% D_A + 5% D_L | Agent-agent interactions |

Factor datasets (D_L, D_A, D_I) created via SAM3 segmentation → tracklet mining → agent/layout separation. D_I still lands on disk at m11 completion; re-enable Stage 3 by adding the yaml block back.

### 🔒 v12 fallback H9 — unfreeze-fraction coarse sweep (if v11 saturates < 3 pp)

The `0.25 / 0.50` fractions come from the FactorJEPA proposal Sec 11.5 (n₁, n₂) + ULMFiT gradual-unfreezing tradition — **not from any V-JEPA 2.1 ViT-G (48-block) ablation**. Literature precedents that informed the ratio (BEiT, DINO-v2, MAE) all operate on ViT-B/L (12-24 blocks); we're 2-4× deeper, so the optimal fraction may differ. Empirical evidence we already have says Stage 3 at 0.75 was *wrong* (BWT=−0.33 at 1K POC → retired), so we know the space is non-trivial.

If v11 (H1 + H4) still fails to clear Δ ≥ 3 pp, the unfreeze fractions become the next-most-plausible suspect. A **1D coordinate sweep** (not a 2D grid) along the proposal's n₁:n₂ = 1:2 diagonal gives a real answer cheaply:

| Config | Stage 1 unfreeze_below | Stage 2 unfreeze_below | Trainable blocks (out of 48) |
|---|---|---|---|
| shallow | 0.20 | 0.40 | 9 → 19 |
| current | 0.25 | 0.50 | 12 → 24 |
| deep | 0.30 | 0.60 | 14 → 28 |

Budget: 3 × ~13 h 10K runs = **~$30 + 40 GPU-h**. Publishable artifact: first "unfreeze depth vs Prec@K at ViT-G scale" curve for V-JEPA 2.1 (methods paper contribution on top of the main result). Caveat: CI ±2.35 pp at N=500 may under-resolve adjacent fractions — if the 3 points overlap in CI, escalate the winning pair to 50K for a decisive reading. Rejected variants (why 1D, not full 2D grid): 2D is 25 runs (~$250 + 300 GPU-h), violates NeurIPS timeline; 1:2 diagonal follows the proposal's symmetry assumption and is defensible in review.

---

## Python Packages with JEPA Training Code

| Package | JEPA Support | Status |
|---------|-------------|--------|
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | **YES** — full training configs in `configs/train/vitg16/` | Active, official |
| [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | **YES** — V-JEPA 1 training (`app/vjepa/train.py`) | Active |
| [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa) | **YES** — lightweight JEPA examples (CIFAR-10, Moving MNIST) | Active (2026) |
| LightlySSL | No JEPA (has BYOL, DINO, SimCLR, MoCo, MAE) | Active |
| solo-learn | No JEPA | Active |
| VISSL | No JEPA | Archived (2024) |

**For Ch10/Ch11**: Use Meta's official `facebookresearch/vjepa2` training code. Configs exist at `configs/train/vitg16/` (2.0) and `configs/train_2_1/vitG16/` (2.1 ablation).

---

## Execution Plan + Historical Results

**Current commands:** 🚀 `iter/iter9/runbook.md` (10K terminal-commands only)
**Current status:** 📋 `iter/iter9/plan_TODO.md` (Step H scale-ladder §)
**Error log:** 🐛 `iter/iter9/errors_N_fixes.md`
**1K POC run log:** 📜 `iter/utils/experiment_log.md` entry 2026-04-19
**Training configs:** ⚙️ `configs/train/` (ch10_pretrain.yaml, explora.yaml, surgery_2stage_noDI.yaml, surgery_2stage_loud_agent.yaml, surgery_3stage_DI.yaml — last 3 renamed from ch11_surgery{,_v15b,_v15c}.yaml on 2026-04-25)

### Historical: 10K POC (DONE ✅)

| Item | Result |
|------|--------|
| Model | V-JEPA 2.0 ViT-g (1B), 1408-dim |
| Data | 10K subset (8,982 train / 1,018 val) |
| Ablation | λ ∈ {0, 0.001, 0.01, 0.1} × 1 epoch each |
| Winner | λ=0.001 (jepa_loss=1.4914, selected by lowest loss) |
| Adapted vs Frozen | Prec@K: 36.14% vs 36.09% (Δ=+0.05%, **noise**) |
| Conclusion | **10K clips insufficient for 1B model adaptation** |

### Step 2a: 115K Full, λ=0.001 — CATASTROPHIC FORGETTING ❌ (2026-04-05)

| Item | Result |
|------|------|
| Model | V-JEPA 2.0 ViT-g (1B), same as POC |
| Data | 115K full corpus (114,576 train / 1K val) |
| Training | 16f, 1 epoch, BS=112, 1023 steps, LR=1e-5, ImageNet norm=YES |
| Eval | 10K POC subset, 64f, BS=44 |
| Lambda | **λ=0.001** |
| JEPA loss | 0.497 → 0.476 (train), best val=1.648 |
| Prec@K | **14.3% adapted vs 36.1% frozen (−21.8pp, significant)** |
| nDCG@K | 0.906 vs 0.950 (−0.045, significant) |
| Diagnosis | λ=0.001 drift penalty (0.00047) is 1000x smaller than JEPA loss (0.476). EWC literature uses λ=10²–10⁹ (arxiv 2505.05946) |
| Full log | `iter/utils/experiment_log.md` |

### Step 2b: λ=100 Ch10 Ablation (PARALLEL, not prerequisite for Ch11)

| Item | Plan |
|------|------|
| Model | V-JEPA 2.0 ViT-g (1B) — same as failed run |
| Data | 115K full corpus, same split |
| Training | 16f, **5 epochs + 1 cooldown**, **LR=1e-6 (constant)**, ImageNet norm=YES |
| Lambda | **λ=100** (100,000x stronger than failed λ=0.001) |
| Anti-forgetting | EWC (FIM-weighted L2) + VICReg variance-covariance |
| Layer freezing | Freeze layers 0-20, train 20-48 only |
| Monitoring | Effective rank + kNN probe → early stop if below frozen baseline |
| Purpose | **Comparison point** — "brute force fails, surgery succeeds" |
| Time | ~6h GPU |

### Step 3: V-JEPA 2.1 (2B) Upgrade — PRIMARY TARGET

V-JEPA 2.1 ViT-G (2B, 1664-dim) is the **primary target model**, not an appendix ablation. Gold standard audit found 2.1's dense loss + deep supervision maximizes spatial feature quality (+23.5 mIoU on ADE20K). Ref: [arXiv:2603.14482](https://arxiv.org/abs/2603.14482)

| Item | V-JEPA 2.0 (current) | V-JEPA 2.1 (target) |
|------|------|------|
| Architecture | ViT-g (1B), standard JEPA | ViT-G (2B), deep self-supervision at 4 intermediate layers |
| Embedding dim | 1408 | 1664 |
| Loss | L1 masked-only | Dense Predictive Loss (ALL tokens, L1) |
| Spatial quality | Baseline | +23.5 mIoU on ADE20K |
| Prerequisite | None | Step 2b validates forgetting control first |

### Step 4: Ch11 Surgery Fine-Tuning — DIRECTLY ON FROZEN

Ch11 runs **directly on the frozen V-JEPA encoder** (no Ch10 prerequisite). See "Key Insight: Ch10 NOT Prerequisite" section below.

---

## Ch10 Training Recipe

**Moved to:** `configs/train/ch10_pretrain.yaml` + `configs/train/base_optimization.yaml`
**Layman explanation:** See "Layman Explanation" in earlier session's plan_training.md commit history.

Key parameters (all in YAML, not hardcoded):
- LR: 1e-6 constant (not cosine), pred_lr 1x (not 10x)
- Lambda: [10, 100, 1000] (was [0, 0.001, 0.01, 0.1] — catastrophic forgetting)
- Dense loss: predict_all=true, lambda_context=0.5
- Deep supervision: 4-layer hierarchical (6656-dim teacher output)
- Freeze layers 0-20, train 20-48
- EWC drift control (FIM-weighted L2)


---

## Gold Standard Audit Fixes (12 Discrepancies Found)

Audit of m09_pretrain.py against V-JEPA 2/2.1 source code and literature (2026-04-10). All CRITICAL/HIGH items must be fixed before next run.

| # | Current | Fix to | Severity | Ref |
|---|---------|--------|----------|-----|
| 1 | Cosine LR decay to 1e-7 | **Single cosine to `min_lr=1e-6`** (build_scheduler always cosine; arXiv:2503.02844 forgetting warning applies to *re-warming*, not single decay; matches Meta continual recipe `final_lr: 0.0`. 2026-04-26: dead `lr_schedule: constant` yaml field deleted via `#78`) | RESOLVED | [arXiv:2503.02844](https://arxiv.org/abs/2503.02844) |
| 2 | V-JEPA 2.0 (1B, 1408d) | **V-JEPA 2.1 (2B, 1664d)** | CRITICAL | [arXiv:2603.14482](https://arxiv.org/abs/2603.14482) |
| 3 | Masked-only L1 loss | **Dense loss (all tokens)** | CRITICAL | V-JEPA 2.1 paper |
| 4 | Final layer supervision | **4-layer deep supervision** | CRITICAL | V-JEPA 2.1 paper |
| 5 | grad_clip=10.0 (V-JEPA 1/2 default for 315-epoch budget) | **1.0** (REVERSED 2026-04-26 via `#78` — at lr=5e-5 over our 1140-step budget, the standard transformer-FT bound bounds outlier batches; Meta's 10.0 is sized for 315-epoch absorbing capacity we don't have) | RESOLVED | BEiT/MAE/DINOv2 FT |
| 6 | 1 epoch | **15 epochs + 1 cooldown** (was 5; bumped 2026-04-26 via `#78` after surgery_2stage_noDI v2 val-loss un-plateaued) | HIGH | [arXiv:2406.14833](https://arxiv.org/abs/2406.14833) |
| 7 | All layers trainable | **Freeze 0-20, train 20-48** | HIGH | [arXiv:2509.10156](https://arxiv.org/abs/2509.10156) |
| 8 | No cooldown phase | **Epoch 6: 64f, linear LR decay** (matches eval frame count) | HIGH | V-JEPA 2 cooldown config |
| 9 | Predictor LR 10x encoder | **Ablate: 10x vs 1x** (predictor = retention mechanism) | HIGH | [arXiv:2311.13321](https://arxiv.org/abs/2311.13321) |
| 10 | Teacher layer_norm missing | **Fixed** | FIXED | V-JEPA 2 train.py line 432 |
| 11 | Uniform L2 drift control | **EWC with FIM-weighted L2** | HIGH | [arXiv:2210.16365](https://arxiv.org/abs/2210.16365), [arXiv:2603.18596](https://arxiv.org/abs/2603.18596) |
| 12 | No collapse prevention | **VICReg variance-covariance term** | HIGH | [arXiv:2410.19560](https://arxiv.org/abs/2410.19560) |

---

## Key Insight: Ch10 is NOT a Prerequisite for Ch11

Ch11's novelty = factor-decomposed inputs + progressive prefix unfreezing using the SAME JEPA loss. This runs **directly on the frozen encoder**. Ch10's adapted checkpoint is not needed.

**Skipping Ch10 makes Ch11's result STRONGER:**

| Approach | What it proves | Paper strength |
|---|---|---|
| Ch10 → Ch11 | "Surgery improves an already-adapted model" | Weak — readers ask "was it Ch10 or Ch11?" |
| **Ch11 directly on frozen** | "Surgery alone fixes what brute-force couldn't" | Strong — clean attribution |
| **Ch11 on frozen + Ch10 as ablation** | "Surgery works AND outperforms brute force" | Strongest — both results |

**Literature supports skipping Ch10:**
- ULMFiT (Howard & Ruder, 2018) — progressive unfreezing directly on pretrained LM
- ExPLoRA ([arXiv:2406.10973](https://arxiv.org/abs/2406.10973), ICML 2025) — LoRA + 2-block unfreezing directly on frozen DINOv2
- LayerLock ([arXiv:2509.10156](https://arxiv.org/abs/2509.10156), ICCV 2025) — progressive freezing during pretraining

| | Skip Ch10 (Ch11 on frozen) | Do Ch10 first |
|---|---|---|
| Time to first result | **Days** | Weeks |
| Attribution | Clean | Confounded |
| Risk | If Ch11 fails, no fallback | Warmer starting point |
| Narrative | "Brute force fails, surgery succeeds" — **strong contrast** | "We pretrained, then refined" — incremental |
| Compute | ~20h | ~100h |
| NeurIPS deadline | Feasible in 3 weeks | Very tight |

**Novel contribution:** No paper addresses JEPA catastrophic forgetting — open research gap. Publishable regardless of result.

---

## Experiment Flow (V-JEPA 2.1, plateau-seeking scale-ladder)

```mermaid
flowchart TD
    POC1K["1K POC (iter8, DONE ❌)<br>N=100 val-split, CI ±4.5 pp<br>Δ +0.17 pp → gate FAILED<br>→ 2-stage recipe + early-stop suite"]

    T10["Tier 10K (iter11 v2, READY ⏳)<br>factor_prep + run_train + run_eval<br>eval_10k paired BCa, N=9,297, CI ±0.42 pp<br>~50-80 h end-to-end"]

    G10{"10K gate<br>Δ vs Frozen?"}
    BWT_B["BWT Option B<br>(λ=50, yaml-only)<br>re-run 10K"]
    BWT_C["BWT Option C<br>(EWC-weighted L2)<br>~3 h impl + re-run"]
    KNEE10["📗 10K is knee<br>submit as-is<br>+ Step G (ExPLoRA)"]
    PIVOT["Pivot →<br>Ch9 diagnostic paper<br>+ temporal-interference"]

    T50["Tier 50K (Step H.1, 🔒)<br>3 TB instance, 55 GB HF pull<br>~68 h worst-case<br>plateau-exit sooner"]
    G50{"50K Δ vs 10K?"}
    KNEE50["📗 50K is knee<br>submit as-is"]

    T115["Tier 115K (Step H.2, 🔒)<br>5 TB instance OR<br>streaming-m11 refactor<br>~98 h worst-case"]
    HEADLINE["📗 115K headline<br>paper asymptote"]

    POC1K --> T10
    T10 --> G10
    G10 -->|"Δ ≥ 2 pp"| T50
    G10 -->|"Δ ∈ [0.5, 2) pp"| BWT_B
    G10 -->|"Δ < 0.5 pp"| KNEE10
    G10 -->|"Δ ≤ 0 pp"| BWT_B
    BWT_B -->|"still ≤ 0"| BWT_C
    BWT_C -->|"still ≤ 0"| PIVOT
    BWT_B -->|"> 0"| T50
    BWT_C -->|"> 0"| T50

    T50 --> G50
    G50 -->|"Δ ≥ 0.5 pp vs 10K"| T115
    G50 -->|"Δ < 0.5 pp vs 10K"| KNEE50

    T115 --> HEADLINE

    style POC1K fill:#546e7a,color:#fff,font-weight:bold
    style T10 fill:#f4511e,color:#fff,font-weight:bold
    style T50 fill:#e53935,color:#fff,font-weight:bold
    style T115 fill:#c62828,color:#fff,font-weight:bold
    style G10 fill:#1976d2,color:#fff,font-weight:bold
    style G50 fill:#1976d2,color:#fff,font-weight:bold
    style BWT_B fill:#8e24aa,color:#fff,font-weight:bold
    style BWT_C fill:#8e24aa,color:#fff,font-weight:bold
    style KNEE10 fill:#43a047,color:#fff,font-weight:bold
    style KNEE50 fill:#43a047,color:#fff,font-weight:bold
    style HEADLINE fill:#2e7d32,color:#fff,font-weight:bold
    style PIVOT fill:#ff8f00,color:#fff,font-weight:bold
```

**Plateau-seeking rationale**: publishable contribution is the scaling *curve* (Prec@K vs log₁₀(n_clips)), not a single number. Each tier = one datapoint with 95 % CI. Early-stop triggers (plateau on val_jepa + negative-BWT + catastrophic kill-switch) auto-halt GPU spend inside a tier → worst-case ≈ plateau-exit case when signal saturates.

## Research Papers: JEPA Family (48 found, 12 most relevant)

### Tier 1: Directly applicable

| Paper | arXiv | Technique | Why it matters |
|---|---|---|---|
| Drive-JEPA | [2601.22032](https://arxiv.org/abs/2601.22032) | V-JEPA continued SSL on driving video | Exact precedent — adjusted BS, WD, LR |
| Surgical V-JEPA | [2509.06831](https://arxiv.org/abs/2509.06831) | V-JEPA continued SSL on surgical video | Validates domain adaptation via continued SSL |
| Beyond Cosine Decay | [2503.02844](https://arxiv.org/abs/2503.02844) | Infinite/constant LR > cosine re-warming | Cosine re-warming causes forgetting |
| LayerLock (ICCV 2025) | [2509.10156](https://arxiv.org/abs/2509.10156) | Progressive freezing for video ViT (4B) | ViT layers converge in depth order |
| EWC for SSL (NeurIPS 2022 WS) | [2210.16365](https://arxiv.org/abs/2210.16365) | EWC works with SSL + ViT | Pre-computed FIM released |

### Tier 2: Informs design

| Paper | arXiv | Technique | Key insight |
|---|---|---|---|
| EWC Done Right | [2603.18596](https://arxiv.org/abs/2603.18596) | FIM gradient vanishing fix | Must read before using EWC |
| JEPA Implicit Bias (NeurIPS 2024) | [2407.03475](https://arxiv.org/abs/2407.03475) | Deeper predictor = robust features | Predictor depth matters |
| Revisiting Supervision (ECCV 2024) | [2311.13321](https://arxiv.org/abs/2311.13321) | MLP projector = retention mechanism | Predictor LR matters for forgetting |
| C-JEPA (NeurIPS 2024) | [2410.19560](https://arxiv.org/abs/2410.19560) | VICReg regularization for JEPA | Prevents collapse during domain shift |
| VJ-VCR | [2412.10925](https://arxiv.org/abs/2412.10925) | Variance-covariance reg for video JEPA | Same setting as ours |
| Stability Gap | [2406.14833](https://arxiv.org/abs/2406.14833) | Multi-epoch > 1 epoch | Recovery requires consolidation time |

### Anti-Forgetting

| Paper | arXiv | Technique |
|---|---|---|
| EWC (original) | [1612.00796](https://arxiv.org/abs/1612.00796) | Fisher Information anchoring |
| GEM | [1706.08840](https://arxiv.org/abs/1706.08840) | Backward Transfer metric + gradient projection |
| Real-time Forgetting Detection | [2512.20634](https://arxiv.org/abs/2512.20634) | Shallow/deep alignment (86-90% accuracy) |
| Dimensional Collapse | [2110.09348](https://arxiv.org/abs/2110.09348) | Effective rank monitoring |
| Same Loss Better Downstream (ICML 2023) | [2210.14199](https://arxiv.org/abs/2210.14199) | SSL loss ≠ downstream. Flatness matters. |
| Progressive SSL Freezing | [2303.07477](https://arxiv.org/abs/2303.07477) | Freeze correlated layers, -1-2% forgetting |

### Transfer Learning (fallback if full-param fails)

| Paper | arXiv | Technique | Forgetting risk |
|---|---|---|---|
| LoRA | [2106.09685](https://arxiv.org/abs/2106.09685) | Low-rank adaptation (~0.1% params) | Near-zero |
| BitFit | [2106.10199](https://arxiv.org/abs/2106.10199) | Bias-only tuning (~0.1% params) | Near-zero |
| Fine-Tuning Distorts | [2202.10054](https://arxiv.org/abs/2202.10054) | Head re-initialization | Low |
| Soft Masking CL | [2302.03241](https://arxiv.org/abs/2302.03241) | Data mixing for continual pretraining | Low |
| Domain-Specific Adapters | [2504.08613](https://arxiv.org/abs/2504.08613) | ViT adapters for domain CL | Low |
| Bayesian Checkpoint Selection | [2410.05612](https://arxiv.org/abs/2410.05612) | Checkpoint quality without labels | N/A (monitoring) |
| Future of CL (survey) | [2506.03320](https://arxiv.org/abs/2506.03320) | SSL = softer updates = less forgetting | N/A (survey) |

### V-JEPA Architecture

| Paper | arXiv | Technique |
|---|---|---|
| V-JEPA (original) | [2404.08471](https://arxiv.org/abs/2404.08471) | Feature prediction in latent space from video |
| V-JEPA 2 | [2506.09985](https://arxiv.org/abs/2506.09985) | 1M+ hours, ViT-G, JEPA + EMA |
| V-JEPA 2.1 | [2603.14482](https://arxiv.org/abs/2603.14482) | Dense loss + deep supervision + 2B model |
| I-JEPA (CVPR 2023) | [2301.08243](https://arxiv.org/abs/2301.08243) | Image JEPA, predictor depth study |

### Techniques Ranked by Expected Impact (with strict goal)

| # | Technique | Impact | Implement? |
|---|---|---|---|
| 1 | V-JEPA 2.1 (2B, dense loss, deep supervision) | CRITICAL | YES |
| 2 | λ = [10, 100, 1000] | CRITICAL | YES |
| 3 | Constant LR (not cosine) | CRITICAL | YES |
| 4 | EWC with FIM (not uniform L2) | HIGH | YES |
| 5 | VICReg variance-covariance | HIGH | YES |
| 6 | Progressive freezing (layers 0-20) | HIGH | YES |
| 7 | 5 epochs + cooldown | HIGH | YES |
| 8 | Effective rank early stopping | HIGH | YES |
| 9 | kNN probe early stopping | HIGH | YES |
| 10 | Predictor LR ablation (10x vs 1x) | MEDIUM | YES |
| 11 | LoRA (fallback) | MEDIUM | IF NEEDED |
| 12 | BitFit (fallback) | LOW | IF NEEDED |

---

## Idea Critic: 7-Dimension Evaluation (Reframed: Temporal Interference Paper)

**Framing:** Temporal interference discovery (Ch9) = the insight. Temporal projection + FactorJEPA surgery (Ch11 on frozen) = the method. Ch10 = ablation comparison.

| # | Dimension | Score | Assessment |
|---|-----------|-------|-----------|
| 1 | **Novelty** | MONTHS (unique combo) | "Temporal encoding corrupts spatial features on OOD data" — no prior work identifies this. Frame shuffling as diagnostic tool is novel. Temporal interference projection is 10 lines of NumPy. Factor-decomposed JEPA surgery is a unique combination. |
| 2 | **Impact** | HIGH | General finding applicable to ANY video foundation model on ANY OOD domain. Not India-specific. Scientific depth: diagnosis + theory + method. |
| 3 | **Timing** | WELL-TIMED | V-JEPA 2.1 just dropped. Geographic bias is hot. Window open but closing. |
| 4 | **Feasibility** | HIGH | Day 1 experiments are CPU-only (30 min + 1h). Ch11 POC = 3h GPU on 100 clips. No Ch10 prerequisite. Feasibility dramatically improved by skipping Ch10 and starting with cheap experiments. |
| 5 | **Competitive** | OPEN | No one doing temporal interference analysis on video SSL. Risk: Meta at 100x scale. Advantage: dataset + diagnostic finding + general theory. |
| 6 | **Nugget** | CLEAR | "Video foundation models suffer temporal interference — temporal features learned from training-domain motion statistics corrupt spatial representations on OOD data. We diagnose it via frame shuffling, remove it via subspace projection, and prevent it via factor-decomposed surgical fine-tuning." |
| 7 | **Narrative** | COMPELLING | (1) Western model fails on India, (2) shuffling IMPROVES results → temporal features are the problem, (3) project out temporal subspace → instant recovery, (4) FactorJEPA surgery prevents it permanently, (5) generalizes to driving + sports + medical. |

### Verdict: PURSUE (upgraded from REFINE)

Temporal interference framing makes this a general contribution, not a dataset paper. Feasibility dramatically improved by skipping Ch10 prerequisite and starting with cheap CPU experiments. No paper addresses JEPA catastrophic forgetting — open research gap, publishable regardless of result.

### Critical Validation (Week 1 — cheap experiments first)

See "Updated Execution Order" section above for the full week-by-week plan.

**Day 1 (CPU):** Temporal interference projection (30 min) + encoder fusion (1h)
**Day 2 (GPU):** Ch11 factor POC directly on frozen (3h)
**Day 3 (GPU):** λ=100 as parallel ablation (6h)
**Day 4-5 (CPU):** Generalize shuffled finding to BDD100K + Diving48

### Decision Tree (updated 2026-04-19, plateau-seeking)

**Per-tier gate** — iter11 v2: paired BCa Δ Prec@K on `data/eval_10k.json` (N=9,297, CI_half ±0.42 pp). iter9 used test_500 (N=500, CI ±2.1 pp); iter11 v2 supersedes with ~5× tighter CI:

| Current tier | Δ (Surgery − Frozen) | Δ vs prior tier | Next action |
|---|---|---|---|
| 10K | ≥ 3 pp | n/a | → 50K (H.1) + Step G (ExPLoRA) |
| 10K | [1, 3) pp | n/a | → BWT Option B (λ=50), re-run 10K |
| 10K | < 1 pp | n/a | → 10K is knee; Step G only, skip ladder |
| 10K | ≤ 0 pp | n/a | → BWT B → C queue, then diagnostic pivot |
| 50K | any | ≥ 0.5 pp over 10K | → 115K (H.2) |
| 50K | any | < 0.5 pp over 10K | → 50K is knee; paper submission tier |
| 115K | any | ≥ 0.5 pp over 50K | → paper headline number |
| 115K | any | < 0.5 pp over 50K | → 50K was asymptote; publishable as "no further benefit" |

Gate thresholds bumped from 2 pp → 3 pp at 10K to account for wider CI (N=500 half-width ±2.1 pp vs ±1.5 pp at N=1000). Non-overlapping 95 % CI requires Δ ≥ 2 × ci_half ≈ 4.2 pp — the 3 pp threshold is partial-non-overlap heuristic; strict reviewer read requires full non-overlap confirmation.

**Paper narrative** (decoupled from scale ladder):
- Surgery > ExPLoRA + projection works → **Strongest NeurIPS: 3 contributions**
- Surgery > ExPLoRA → **Strong NeurIPS: factor surgery beats SOTA adaptation**
- ExPLoRA improves, surgery = ExPLoRA → **Publish ExPLoRA-on-V-JEPA-2.1, surgery as ablation**
- Nothing improves Prec@K → **Submit Ch9 diagnostic + temporal interference finding**

---

## Best Paper Strategy (Brainstormer Reframing)

### The One-Sentence Paper

> "Video foundation models trained on Western data suffer from temporal interference — temporal features learned from Western motion statistics actively corrupt spatial representations on out-of-distribution domains — and we show that this interference can be diagnosed via frame shuffling, removed via subspace projection, and prevented via factor-decomposed surgical fine-tuning."

### Why the Current Proposal Won't Win Best Paper

Best Papers at NeurIPS share three traits: (1) a **surprising, general insight**, (2) a **simple, elegant method** that follows from the insight, (3) **thorough experiments** that prove generality beyond one dataset.

Current proposal has trait (1) buried in Ch9, lacks trait (2) because Ch10+Ch11 is a 12-variable kitchen sink, and lacks trait (3) because everything is on one dataset.

### Key Reframing

> V-JEPA's spatial features for Indian scenes are NOT missing — they're being CORRUPTED by temporal features. The shuffled > normal result proves this. The fix should REMOVE the corruption, not RETRAIN the whole model.

| | Current Proposal | Best Paper Version |
|---|---|---|
| **Title** | "FactorJEPA: Factor-Decomposed Surgical Fine-Tuning" | "Temporal Interference in Video Foundation Models: Diagnosis, Theory, and Surgery" |
| **Insight** | "Indian streets need domain adaptation" (specific) | "Temporal encoding actively corrupts spatial features when motion statistics shift" (general) |
| **Method** | 12-variable training recipe | Temporal interference projection (10 lines NumPy) + FactorJEPA when projection isn't enough |
| **Scope** | Indian streets only | Any domain where motion statistics differ from training |
| **Generality** | None tested | Test on driving (BDD100K), sports (Diving48), medical (surgical) |

### Restructured Paper: 5 Contributions

**Contribution 1 (Ch9): Diagnosis — The Temporal Interference Discovery**
- Shuffled V-JEPA > normal V-JEPA by 2.4x on Indian streets
- NOT a dataset bug — evidence of a systematic failure mode
- **Generalize:** test shuffled vs normal on 2-3 other OOD domains. If shuffling helps on ALL → general phenomenon, not a quirk of Indian data.

**Contribution 2 (NEW): Theory — The Temporal Interference Subspace**
- PCA on (normal_embedding - shuffled_embedding) for 10K clips
- Top principal components = the temporal interference subspace
- Project embeddings orthogonally → measure Prec@K recovery
- **30-minute CPU experiment that could be the paper's centerpiece**
- If it works: "We identify and remove the temporal interference subspace, recovering X% Prec@K with zero retraining"

**Contribution 3 (Ch10): Baseline — Continual Pretraining (What Doesn't Work)**
- Naive continual pretraining → catastrophic forgetting
- EWC + proper λ prevents forgetting but doesn't improve spatial features
- The "negative result that motivates the real solution"

**Contribution 4 (Ch11): Method — FactorJEPA (What Does Work)**
- Factor decomposition into layout/agent/interaction via SAM3
- Progressive prefix unfreezing
- Same JEPA loss — only input distribution and trainable depth change
- Key comparison: Ch10 (brute force) vs Ch11 (surgical) on same loss curve

**Contribution 5 (NEW): The Complementary Strengths Result**
- V-JEPA dominates temporal (Cycle@K 78.7%), DINOv2 dominates spatial (Prec@K 50.5%)
- Weighted fusion alpha*V-JEPA + (1-alpha)*DINOv2 outperforms both on ALL metrics
- 1-hour experiment, publishable baseline

### References from Brainstormer

| Paper | arXiv | Relevance |
|---|---|---|
| Drift-Adapter (EMNLP 2025) | [2509.23471](https://arxiv.org/abs/2509.23471) | Affine map between embedding spaces, 95-99% recall recovery |
| Representation Surgery (ICML 2024) | [2402.09631](https://arxiv.org/abs/2402.09631) | Optimal affine steering to remove harmful subspaces |
| ExPLoRA (ICML 2025) | [2406.10973](https://arxiv.org/abs/2406.10973) | LoRA + 2-block unfreezing for DINOv2 domain adaptation |
| Difference-Masking (EMNLP 2023) | [2305.14577](https://arxiv.org/abs/2305.14577) | Preferentially mask domain-specific regions during pretraining |
| Temporal vs Spatial (arXiv) | [2509.21595](https://arxiv.org/abs/2509.21595) | Confirms DINOv3 > V-JEPA spatial tradeoff independently |
