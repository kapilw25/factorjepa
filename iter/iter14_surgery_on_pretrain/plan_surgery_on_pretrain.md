# 🎯 iter14 — Surgery on Pretrain Q&A

> **Paper goal**: `vjepa_surgery` ≫ `vjepa_pretrain` ≫ `vjepa_frozen` on motion / temporal features
>
> **iter14 hypothesis**: surgery trained ON TOP OF pretrain (sequential SSL composition) outperforms surgery from frozen V-JEPA, AND outperforms a compute-matched long-pretrain control — proving the gain is from factor patching, not from extra training steps.

---

## 📊 Empirical anchor (already in the bag — `iter/iter13_motion_probe_eval/`)

| Metric | Frozen | Pretrain (5ep) | Δ vs Frozen |
|---|---:|---:|---|
| 🎯 `probe_top1` (motion-flow 16-class) | — | **0.808** | monotonic ↑ from 0.439 over 5 epochs |
| 🌀 `motion_cos` (intra-vs-inter cosine) | 0.046 (init) | **0.267** | **5.8×** — key paper signal |
| 🔮 `future_mse` | 0.5571 (CI 0.5561–0.5581) | **0.5544** (CI 0.5531–0.5557) | **Δ = +0.0027, p = 0.0** ✅ |
| 📉 `val_jepa_loss` | 0.473 (init) | 0.458 | ↓ 3.2 % |
| 🧱 `block_drift_mean` | 0.0 | 0.0160 | healthy (1.6 %) |
| 📏 `‖Δ‖/‖init‖` | 0.0 | **2.46 %** | non-collapsed, non-stuck |

🟢 **Half the strict ordering is already proven**: `pretrain > frozen` on `future_mse` with non-overlapping 95 % CI.

---

## ❓ Q1 — Preserving pretrain gains under surgery

### 🩺 Q1.1 — How to **MONITOR** loss of pretrain gains?

> 🛡️ Per-layer **CKA similarity** vs `θ^(pretrain)` (already partially wired via `m09a_block_drift.png`); held-out **general-video probe** accuracy (K400 / SSv2 retention — **NOT yet tracked**); `val_jepa_loss` on a frozen pretrain-distribution validation slice; **gradient-norm spikes** per stage; **weight-norm trajectory**; an "**old probe**" (frozen probe trained on pretrain features, applied to surgery checkpoints — drop = forgetting).

### 🔧 Q1.2 — Other measures to **PRESERVE** pretrain gains (beyond drift control + LR decay)?

| # | Measure | What |
|---|---|---|
| 🔁 (c) | **Replay** | mix 5–10 % pretrain-distribution batches into surgery |
| 📐 (d) | **EMA / weight averaging** | high `τ ≥ 0.99` so teacher tracks student slowly |
| 🎓 (e) | **KL distillation** | aux loss = KL(student_logits ‖ pretrain_logits) |
| 🪶 (f) | **LoRA / adapter-only** | freeze base, train low-rank delta |
| 🧮 (g) | **Elastic Weight Consolidation** | `λ Σᵢ Fᵢ(θᵢ − θᵢ^pretrain)²` (Fisher-weighted) |
| ⚓ (h) | **L2 anchor loss** | `λ ‖θ − θ_pretrain‖²` (proposal Sec 10.6 already has for frozen→pretrain — reuse for pretrain→surgery) |
| 🛑 (i) | **Early stopping** | abort if pretrain-val loss rises |

### 🔥 Q1.3 — Handling Stage-1 **catastrophic forgetting** if backbone LR is too high?

- 🚦 Cap backbone LR at **≤ 1e-5** (vs predictor 1e-4); 
- short **100–500-step warmup** at each stage boundary (proposal Sec 11.6 has this); 
- **EMA `τ ≥ 0.99`**; **layer-wise LR decay 0.7–0.9** across the unfrozen prefix; 
- **anchor loss** `λ‖θ−θ_pretrain‖²` with `λ ∈ [0.001, 0.01]`; ⚠️ **early-abort surgery Stage-1** if `val_jepa` on pretrain-val rises **> 5 %**.

---

## ❓ Q2 — Reuse the pretrain checkpoint?

### ✅ Q2.1 — Is `m09a_pretrain` checkpoint good to reuse for surgery init?

> **YES** ✅ — `probe_top1` 0.808 (peaked at last step, no plateau yet), `val_jepa` ↓ 3.2 %, `motion_cos` ↑ 5.8×, `block_drift` healthy at 1.6 % mean, `‖Δ‖/‖init‖ = 2.46 %` (non-collapsed, non-stuck), and **`pretrain > frozen` on `future_mse` with non-overlapping 95 % CI** (Δ = +0.0027, p = 0.0). Surgery built on top of this checkpoint is a sound foundation.

### 🤗 Q2.2 — Should we push pretrain to HF?

> **YES** ✅ — push:
> - 📦 `outputs/full/probe_pretrain/student_encoder.pt` (6.9 GB; the inference-ready ViT-G weights — **skip** the 14 GB `m09a_ckpt_best.pt` since it's optimizer-state-bearing and only needed for resume)
> - 📈 `training_summary.json` + `probe_history.jsonl`
> - 🖼️ The 8 plot PDFs
>
> Use the existing `python -u src/utils/hf_outputs.py upload outputs/full` (already wired with `_mirror_cleanup` + `_stale_checkpoint_ignores`). Add a `_generate_model_card`-style README mirroring `cita_ecliptica/push_automation.py` with metrics: arch=V-JEPA 2.1 ViT-G · base=`facebook/v-jepa-2-vitg` · 5 epochs · 32,320 clips · `probe_top1=0.808` · `motion_cos↑5.8×` · `future_mse Δ=+0.27 % over frozen, p=0.0`.

---

## ❓ Q3 — Compute-matched long-pretrain ablation

### 🧮 Q3.1 — Compute-matched control explained with numbers

> **YES**, exactly the framing:
>
> | Arm | Composition | Total budget | Factor patching? |
> |---|---|---|---|
> | **A — current pretrain** | pretrain (5 ep / 1,010 steps / ~10 GPU-h) | 5 ep | ❌ |
> | **B — surgery on pretrain** | pretrain (5 ep) ▶ surgery (5 ep) | **10 ep / ~20 GPU-h** | ✅ in last 5 ep |
> | **C — long-pretrain control** | pretrain (10 ep / 2,020 steps / ~20 GPU-h) | **10 ep / ~20 GPU-h** | ❌ |
>
> 🎯 **The proof**: if `B > C` with non-overlapping 95 % CI on `motion_cos` / `future_mse` / `probe_top1`, the gain is **causally attributed to factor patching** rather than the extra steps.
>
> 💰 Cost: **one additional ~10 GPU-h training run** (vs current pretrain).

### 🤖 Q3.2 — Is "compute-matched extended-SFT" common in RL/RLHF papers?

> **YES** — Tülu 3 (Lambert et al., 2024) and *"Is DPO Superior to PPO?"* (Xu et al., 2024 OpenReview) both include **compute-matched extended-SFT baselines** as ablation arms. The "2×" isn't a magic ratio — the canonical pattern is *"match total compute = SFT_steps + (DPO|PPO)_steps"*. InstructGPT / RLAIF papers use the same convention via best-of-N sampling from prolonged SFT as a control.

---

## 📚 Sources

- 🎬 [V-JEPA 2: Self-Supervised Video Models (Assran et al., 2025)](https://arxiv.org/abs/2506.09985) — staged training pattern + EMA momentum schedule
- 🎬 [V-JEPA 2 Meta AI page](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) — official architectural description
- 🔗 [Two-Stage Fine-Tuning Strategy survey](https://www.emergentmind.com/topics/two-stage-fine-tuning-strategy) — sequential stages outperform interleaved; LR + replay are canonical preservation tools
- 🔗 [Sequential Finetuning (SeqL)](https://www.emergentmind.com/topics/sequential-finetuning-seql) — sequential composition outperforms single-stage joint training
- 🧠 [Comprehensive Survey on Continual Learning in Generative Models (2025)](https://www.arxiv.org/pdf/2506.13045v2) — EWC, replay, KL distillation, layer-wise LR decay as canonical anti-forgetting
- 🥊 [Is DPO Superior to PPO for LLM Alignment? (Xu et al., 2024 OpenReview)](https://openreview.net/pdf?id=6XH8R7YrSk) — compute-matched SFT baseline ablation
- 🛠️ [How to align open LLMs in 2025 with DPO (Schmid)](https://www.philschmid.de/rl-with-llms-in-2025-dpo) — extended-SFT baseline patterns
- 📖 [Direct Preference Optimization tutorial (HF blog)](https://huggingface.co/blog/ariG23498/rlhf-to-dpo) — SFT → DPO sequential composition canonical recipe

▎ Surgery epoch budget — pick ONE:                                                                                        
▎                                                                                                                       
▎ | Option                                             | Surgery epochs                 | Long-pretrain control | Total   
▎ iter14 GPU-h | Why                                                      |     
▎ |----------------------------------------------------|--------------------------------|-----------------------|---------
▎ -----------|----------------------------------------------------------|
▎ | 🅰️  "5+5" (matches your Q3.1 framing)               | 5 ep total (stage split 2/2/1) | 10 ep                 | ~41 h   
▎            | Cheap; matches phrasing in plan_surgery_on_pretrain.md   |                                                 
▎ | 🅱️  "5+15" (keeps current surgery_base.yaml 3×5=15) | 15 ep total (5/5/5 default)    | 20 ep                 | ~71 h   
▎            | Surgery gets full convergence per stage; control 2× cost |                                                 
▎                                                                                                                         
▎ 👉 Recommend 🅰️  — the cheapest experiment that still answers the causal question. If 🅰️  shows null result, escalate to  
▎ 🅱️ .                                                                                                                     
                                                                                                                        
The plan below assumes 🅰️ .                                     
                                                                                                                        
---                                                            
📋 Plan A — 7 file edits, in dependency order                                                                             
                                            
1️⃣  NEW: configs/train/probe_pretrain_long.yaml (~12 LoC)                                                                  
                                            
# iter14 — compute-matched control for `surgery(5+5)` vs `long-pretrain(10)`                                              
# Identical to probe_pretrain.yaml except max_epochs.full doubled 5→10.
# Ablation: NO factor patching, NO surgery — just longer continual SSL.
extends: probe_pretrain.yaml                                        
                                                                    
optimization:                                                                                                             
max_epochs:                                                     
    sanity: 1     # unchanged — code-path validation                                                                      
    poc: 2        # unchanged                                                                                             
    full: 10      # iter14 — was 5 in probe_pretrain.yaml                                                                 
                                                                                                                        
Why a sibling yaml not flag override: keeps probe_pretrain.yaml (the proven 5-ep recipe) immutable so iter13's 5-ep run   
remains reproducible. New yaml is a strict superset — same hyperparameters, just longer.                                  
                                                                                                                        
---                                                                                                                       
2️⃣  NEW: configs/train/surgery_3stage_DI_iter14.yaml (~10 LoC)     
                                                                                                                        
# iter14 — surgery with 5-epoch total budget (was 15 in surgery_base via base_optimization)                               
# Stage split: 2 / 2 / 1 epochs (D_L → D_A → D_I, decreasing as deeper layers unfreeze).                                  
extends: surgery_3stage_DI.yaml                                                                                           
                                                                                                                        
optimization:                                                                                                             
max_epochs:                                                     
    sanity: 1                                                     
    poc: 1                                                                                                                
    full: 5         # iter14 — was 15                               
                                                                                                                        
# iter14 anti-forgetting belt-and-braces (Q1.2 / Q1.3)                                                                    
drift_control:                                                                                                            
lambda_reg: 0.005    # was 0.0 in surgery_base post-iter13-v12; reinstated for iter14 sequential composition            
anchor_to: pretrain  # NEW key — read by m09c (see step 4 below); anchor target = the loaded init ckpt
                                                                    
Mirror file for surgery_2stage_noDI_iter14.yaml.                                                                          
                                                                    
Why these new yamls: literature (Q1.3) flags that without λ ≥ 0.001 anchor loss, surgery Stage-1 with even modestly high  
LR can overwrite pretrain gains. iter13 ran with λ=0.0 from frozen init — fine, no gains to preserve. iter14 has gains to 
preserve.                                                         
                                                                                                                        
---                                                                                                                       
3️⃣  EDIT: src/m09c_surgery.py (~25 LoC)                                                                                    
                                                                                                                        
Current behavior (lines 259–276): m09c always loads the V-JEPA 2.1 frozen ckpt from URL or local path, regardless of      
whether a previous training run produced a better starting point.                                                         
                                                                    
# CURRENT (m09c_surgery.py:259-276) — loads frozen V-JEPA always                                                          
print(f"Downloading pretrained weights: {ckpt_url}")           
ckpt = torch.hub.load_state_dict_from_url(ckpt_url, ...)            
if "target_encoder" in ckpt:                                        
    state_dict = ckpt["target_encoder"]   # uses EMA teacher of frozen                                                    
elif "encoder" in ckpt:                                           
    ...                                                                                                                   
msg = student.load_state_dict(state_dict, strict=False)             
                                                                                                                        
Add a CLI flag + branch:                                                                                                  
                                                                                                                        
# NEW — argparse block (near other --encoder-ckpt args, ~line 1370)                                                       
parser.add_argument(                                                                                                      
    "--init-from-ckpt", default=None,                                                                                     
    help="iter14: load student weights from a prior training-run checkpoint "                                             
        "(e.g. outputs/full/m09a_pretrain/student_encoder.pt) INSTEAD of "                                               
        "the frozen V-JEPA URL. Enables sequential SSL composition "                                                     
        "(pretrain → surgery). When unset, falls back to legacy frozen-init."
)                                                                                                                         
                                                                    
# NEW — load_pretrained() function (replaces lines 259-276)                                                               
def _load_init_state(ckpt_url_or_path, init_from_ckpt=None):        
    """iter14: prefer --init-from-ckpt over the legacy frozen URL when provided."""                                       
    if init_from_ckpt is not None:                                                                                        
        path = Path(init_from_ckpt)                                                                                       
        assert path.exists(), f"--init-from-ckpt missing: {path}"                                                         
        print(f"  [iter14] Loading init from prior-run ckpt: {path}")                                                     
        ckpt = torch.load(path, map_location="cpu", weights_only=False)                                                   
        # student_encoder.pt is exported flat (state_dict at top level OR under 'state_dict')                             
        return ckpt.get("state_dict", ckpt)                       
    # Legacy frozen path                                                                                                  
    print(f"Downloading pretrained weights: {ckpt_url_or_path}")                                                          
    ckpt = torch.hub.load_state_dict_from_url(ckpt_url_or_path, map_location="cpu", weights_only=False)                   
    if "target_encoder" in ckpt: return ckpt["target_encoder"]                                                            
    if "encoder" in ckpt: return ckpt["encoder"]                    
    return ckpt                                                                                                           
                                                                    
# NEW — anchor loss reads drift_control.anchor_to from yaml + caches θ_pretrain at init                                   
# (~10 LoC inserted into the train_step around the existing JEPA loss)                                                    
if drift_cfg.get("anchor_to") == "pretrain" and theta_pretrain is not None:                                               
    anchor_loss = sum(((p - p0) ** 2).sum()                                                                               
                    for p, p0 in zip(student.parameters(), theta_pretrain))                                             
    total_loss = jepa_loss + drift_cfg["lambda_reg"] * anchor_loss                                                        
                                                                                                                        
Why minimal: re-uses the existing lambda_reg plumbing (already in surgery_base.yaml). Only the anchor target changes — was
θ^(0) (frozen V-JEPA), becomes θ_pretrain (continual SSL student).                                                       
                                                                
---                                                                                                                       
4️⃣  EDIT: scripts/run_probe_train.sh (~30 LoC)                                                                             
                                                                                                                        
Add a 4th subcommand pretrain_long + thread --init-from-ckpt through surgery dispatch.                                    
                                                                                                                        
# CURRENT case "$SUBCMD" already accepts pretrain | surgery_3stage_DI | surgery_noDI                                      
                                                                
# NEW case branch (mirror of pretrain, different yaml + output dir):                                                      
pretrain_long)                                                      
    OUT_DIR="outputs/${mode_dir}/m09a_pretrain_long"           
    TRAIN_CFG="configs/train/probe_pretrain_long.yaml"                                                                    
    LAMBDA_REG=$(scripts/lib/yaml_extract.py "$TRAIN_CFG" drift_control.lambda_reg)
    echo "═══ iter14 long-pretrain (compute-matched control, ${MODE}) ═══"                                                
    # ... identical body to pretrain branch, just substitute OUT_DIR + TRAIN_CFG ...
    python -u src/m09a_pretrain.py "${MODE_FLAG}" \                                                                       
        --model-config "$MODEL_CFG" --train-config "$TRAIN_CFG" \                                                         
        --subset "$TRAIN_SPLIT" --val-subset "$VAL_SPLIT" \                                                               
        --local-data "$LOCAL_DATA" --val-local-data "$LOCAL_DATA" \                                                       
        --output-dir "$OUT_DIR" \                                                                                         
        --cache-policy "$P_M09" --lambda-reg "$LAMBDA_REG" \                                                              
        --probe-action-labels "outputs/${mode_dir}/probe_action/action_labels.json" \                                     
        --motion-features-path "${LOCAL_DATA}/motion_features.npy" \                                                      
        "${TAXONOMY_ARGS[@]}" --no-wandb \                                                                                
        2>&1 | tee "logs/m09a_pretrain_long_${mode_dir}.log"                                                              
    ;;                                                                                                                    
                                                                                                                        
# EDIT existing surgery_3stage_DI / surgery_noDI branches:                                                                
# Auto-detect pretrain ckpt; pass via --init-from-ckpt iff present.                                                       
PRETRAIN_CKPT="outputs/${mode_dir}/m09a_pretrain/student_encoder.pt"                                                      
INIT_FLAG=""                                                                                                              
if [ -f "$PRETRAIN_CKPT" ]; then                                                                                          
    echo "  [iter14] surgery will init from pretrain: $PRETRAIN_CKPT"                                                     
    INIT_FLAG="--init-from-ckpt $PRETRAIN_CKPT"                                                                           
else                                                                                                                      
    echo "  [iter14] WARN: $PRETRAIN_CKPT missing — surgery falls back to frozen V-JEPA init"                             
fi                                                             
# (then add $INIT_FLAG to the python -u src/m09c_surgery.py invocation)                                                   
                                                                                                                        
# Also point surgery to the iter14 yamls:                                                                                 
case "$SUBCMD" in                                                                                                         
    surgery_3stage_DI) TRAIN_CFG="configs/train/surgery_3stage_DI_iter14.yaml" ;;                                         
    surgery_noDI)      TRAIN_CFG="configs/train/surgery_2stage_noDI_iter14.yaml" ;;                                       
esac                                                                                                                      
                                                                                                                        
Validation: argparse case statement at the top accepts pretrain_long as a 4th valid subcommand.
                                                                                                                        
---                                                                                                                       
5️⃣  EDIT: scripts/run_probe_eval.sh (~15 LoC)                                                                              
                                                                                                                        
Add vjepa_2_1_pretrain_long to the default ENCODERS set + per-encoder ckpt resolvers.                                     
                                                                                                                        
# Default ENCODERS (line 143):                                                                                            
ENCODERS="${ENCODERS:-vjepa_2_1_frozen vjepa_2_1_pretrain vjepa_2_1_pretrain_long vjepa_2_1_surgical_3stage_DI            
vjepa_2_1_surgical_noDI}"                                           
                                                                                                                        
# encoder_ckpt_for() — add case (around line 178):             
vjepa_2_1_pretrain_long)                                                                                                  
    echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_long/student_encoder.pt" ;;
                                                                                                                        
# encoder_predictor_ckpt_for() — add case (around line 188):        
vjepa_2_1_pretrain_long)                                                                                                  
    echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_long/m09a_ckpt_best.pt" ;;                                               
                                                                                                                        
# pretrain_cleanup_get_latest() — add case (~line 404):                                                                   
vjepa_2_1_pretrain_long)                                            
    echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_long/m09a_ckpt_latest.pt" ;;                                             
                                                                                                                        
# Pre-flight failure messaging (~line 322): add ./scripts/run_probe_train.sh pretrain_long --$MODE hint                   
                                                                                                                        
The existing pre-flight (lines 304–340) ALREADY drops missing-ckpt encoders silently and continues — so if pretrain_long  
hasn't run yet, eval still works on the other 4 encoders. Zero risk of breaking iter13's working pipeline.                
                                                                                                                        
---                                                                                                                       
6️⃣  EDIT: src/probe_action.py --stage paired_delta (~20 LoC)                                                               
                                                                                                                        
Currently emits all O(N²) pairs unlabeled. Add explicit Δ1/Δ2/Δ3 keys to the JSON output.                                 
                                                                                                                        
# In the paired_delta stage (after computing all pairs):                                                                  
ITER14_DELTAS = [                                                 
    ("delta_1_pretrain_vs_frozen",                                                                                        
    "vjepa_2_1_pretrain", "vjepa_2_1_frozen",                      
    "Δ1: continual SSL > frozen (proves domain adaptation works)"),
    ("delta_2_surgical_vs_pretrain",                                                                                      
    "vjepa_2_1_surgical_3stage_DI", "vjepa_2_1_pretrain",          
    "Δ2: surgery > pretrain (proves factor patching adds value)"),                                                       
    ("delta_3_surgical_vs_pretrain_long",                           
    "vjepa_2_1_surgical_3stage_DI", "vjepa_2_1_pretrain_long",                                                           
    "Δ3: surgery > long-pretrain (proves factor patching is CAUSAL, not extra steps)"),                                  
]                                                                                                                         
out["iter14_paper_deltas"] = {}                                                                                           
for key, a, b, desc in ITER14_DELTAS:                                                                                     
    if a in encoder_metrics and b in encoder_metrics:                                                                     
        delta = compute_paired_bca(encoder_metrics[a], encoder_metrics[b], n_resamples=10000)                             
        delta["interpretation"] = desc                              
        delta["pass"] = (delta["ci_lo"] > 0)   # non-overlapping 95% CI ⇒ significant                                     
        out["iter14_paper_deltas"][key] = delta                                                                           
                                                                                                                        
This makes the paper claim self-documenting in probe_paired_delta.json.                                                   
                                                                                                                        
---                                                               
7️⃣  NEW: iter/iter14_surgery_on_pretrain/runbook.md (~50 LoC)                                                              
                                                                                                                        
Just the canonical command sequence; no logic. (Keeps the high-level plan free of bash.)                                  
                                                                                                                        
# === iter14 execution order ===                                    
                                                                                                                        
# 0. Verify prerequisites                                      
ls outputs/full/m09a_pretrain/student_encoder.pt    # ← from iter13 (6.9 GB)                                              
ls data/eval_10k_local/m11_factor_datasets/D_L      # ← factor data
                                                                                                                        
# 1. Surgery (5+5) — runs FIRST; if Δ2 fails, we abort long-pretrain to save GPU
tmux new -s iter14_s                                                                                                      
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL \
    2>&1 | tee logs/iter14_surgery_3stage_DI.log         # ~10 GPU-h                                                      
                                                                                                                        
# 2. Surgery_noDI (ablation) — same init, narrower factor set       
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI --FULL \                                                     
    2>&1 | tee logs/iter14_surgery_noDI.log              # ~7 GPU-h                                                       
                                                                                                                        
# 3. Long-pretrain (10 ep) — control. ONLY run if surgery shows signal in eval.                                           
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain_long --FULL \                                                    
    2>&1 | tee logs/iter14_pretrain_long.log             # ~20 GPU-h                                                      
                                                                                                                        
# 4. 4-encoder eval (frozen + pretrain + pretrain_long + surgery + surgery_noDI)                                          
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL \                                                                   
    2>&1 | tee logs/iter14_probe_eval.log                # ~4 GPU-h                                                       
                                                                                                                        
# 5. Inspect Δ1/Δ2/Δ3 in the new schema                                                                                   
jq '.iter14_paper_deltas' outputs/full/probe_action/probe_paired_delta.json                                               
                                                                                                                        
---                                                                                                                       
✅ Verification gates (per CLAUDE.md "VERIFY-FIRST")                                                                      
                                                                                                                        
┌─────────────┬────────────────────────────────────────────────────────────────────────┬──────────────────────────────┐   
│    Stage    │                                  Gate                                  │       Action if fails        │
├─────────────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────┤   
│ After step  │ outputs/full/m09c_surgery_3stage_DI/student_encoder.pt exists;         │ check anchor_loss is firing; │
│ 1 (surgery) │ probe_top1 in m09c_pretrain_log.csv is ≥ 0.808 (not regressing vs      │  bump λ to 0.01              │   
│             │ pretrain)                                                              │                              │
├─────────────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────┤   
│ After step  │ block_drift_mean in surgery's last checkpoint < 0.05                   │ LR too high; cap backbone LR │   
│ 1           │                                                                        │  ≤ 1e-5                      │
├─────────────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────┤   
│ After step  │                                                                        │ Δ2 fail = abort              │
│ 4 (eval)    │ iter14_paper_deltas.delta_2.ci_lo > 0                                  │ long-pretrain; report        │
│             │                                                                        │ negative result              │
├─────────────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────┤   
│             │                                                                        │ Δ3 fail with Δ2 pass =       │
│ After step  │ iter14_paper_deltas.delta_3.ci_lo > 0                                  │ weaker claim (factor         │   
│ 4           │                                                                        │ patching ≥ extra steps);     │   
│             │                                                                        │ still publishable            │
└─────────────┴────────────────────────────────────────────────────────────────────────┴──────────────────────────────┘   
                                                                                                                        
---            
🧪 Single end-to-end SANITY smoke before FULL                                                                             
                                                                                                                        
# ~25 min total, 24 GB or 96 GB                                     
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain_long --SANITY 2>&1 | tee logs/iter14_sanity_pretrain_long.log    
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --SANITY 2>&1 | tee logs/iter14_sanity_surgery.log
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --sanity 2>&1 | tee logs/iter14_sanity_eval.log
                                                                    
This validates the new --init-from-ckpt wiring + new yaml extends + new encoder resolvers + new Δ1/Δ2/Δ3 emission, all on
24 GB before committing 41 GPU-h to FULL.                           
                                                                                                                        
---                                                               
❓ Three approval-gates before I touch any code                                                                           
                                                                                                                        
1. Epoch budget — confirm 🅰️  "5+5 vs 10" (cheap path, ~$33). Or 🅱️  "5+15 vs 20" (~$57)?                                   
2. Anchor λ — go with λ = 0.005 (literature default), or 3-point sweep {0.001, 0.005, 0.01} (3× surgery cost)?            
3. HF push of pretrain — push student_encoder.pt to HF BEFORE iter14 starts (so surgery downloads from HF for             
reproducibility), or after iter14 done (single bundle)?                                                                   
                                                                                                                        
Say "go with 🅰️ , λ=0.005, push later" (or your variant) and I'll execute the 7 file edits + run the SANITY smoke.
                                                                    
---                                                                                                                       
📋 New TODO state                                              
                                                                                                                        
┌─────┬────────────────────────────────────────────────────────────────────┬────────────────────────────────┐             
│  #  │                                Task                                │             Status             │             
├─────┼────────────────────────────────────────────────────────────────────┼────────────────────────────────┤             
│ 185 │ T1 — Refactor plan_surgery_on_pretrain.md as Q&A                   │ ✅ done                        │             
├─────┼────────────────────────────────────────────────────────────────────┼────────────────────────────────┤
│ 186 │ T2 — Prune + rewrite plan_HIGH_LEVEL.md for iter14                 │ ✅ done                        │             
├─────┼────────────────────────────────────────────────────────────────────┼────────────────────────────────┤
│ 187 │ T3 — PLAN-only: code changes for surgery(5+5) vs long-pretrain(10) │ 🔄 in_progress (this response) │
└─────┴────────────────────────────────────────────────────────────────────┴────────────────────────────────┘
                                                                                                                        
After your approvals on the 3 gates above, I'll create new tasks for T4 (code edits — 7 files), T5 (SANITY smoke), T6
(FULL surgery+surgery_noDI), T7 (FULL long-pretrain — gated on Δ2 signal), T8 (FULL eval), T9 (paper-claim Δ1/Δ2/Δ3       
reporting).  