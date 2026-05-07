# 🎯 iter14 — Surgery on Pretrain · Q&A + Code Plan

> **Paper goal**: `vjepa_surgery` ≫ `vjepa_pretrain` ≫ `vjepa_frozen` on motion / temporal features
>
> **iter14 hypothesis**: surgery built ON TOP OF pretrain (sequential SSL composition) outperforms a compute-matched long-pretrain control — proving the gain is from **factor patching**, not from extra training steps.

---

## 🌅 Next-day pickup (state as of 2026-05-08 EOD)

| 🚦 | Item | Detail |
|---|---|---|
| ✅ | **HF endpoint live** | `https://huggingface.co/anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep` (22.4 GB · `student_encoder.pt` 7.4G + `m09a_ckpt_best.pt` 15G + plots/JSONs) |
| ✅ | **SANITY 1** (probe_eval Stage 8 / `predictor` key) | passed |
| ✅ | **SANITY 2** (surgery init / 1.84 B params · 588 keys via `student_state_dict`) | passed |
| ✅ | **Plan documented** | this file + `plan_HIGH_LEVEL.md` |
| 🟡 | **BLOCKED** — 3 approval gates needed before code edits | see [§ Three gates](#-three-approval-gates-before-touching-code) |
| ⏭️ | **Next action** | execute T4 (7 file edits — see [§ T3 code plan](#%EF%B8%8F-t3-code-plan--7-file-edits)) |

### 🔑 Resume kit (paste-ready when you SSH back in)

```bash
# 0. activate venv + cd
cd /workspace/factorjepa && source venv_walkindia/bin/activate

# 1. verify HF endpoint (uses HF_TOKEN from .env, ~1s if cached)
python -c "from dotenv import load_dotenv; load_dotenv(); from huggingface_hub import hf_hub_download; import torch; p = hf_hub_download('anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep', 'student_encoder.pt'); s = torch.load(p, map_location='cpu', weights_only=False); n = sum(t.numel() for t in s['student_state_dict'].values()); print(f'✅ HF endpoint live · {n:,} params · {len(s[\"student_state_dict\"])} keys')"

# 2. verify local pretrain artifact still there (in case GPU instance preserves /workspace)
ls -lh outputs/full/probe_pretrain/{student_encoder.pt,m09a_ckpt_best.pt} 2>/dev/null || echo "↑ MISSING — re-download via hf_hub_download"

# 3. inspect TODO state
# task IDs: 188=T4 code execute · 189=T5 SANITY · 190=T6 FULL
```

### 📂 Files of interest (clickable in your editor)

- `iter/iter14_surgery_on_pretrain/plan_HIGH_LEVEL.md` — 4-arm experimental design + decision matrix
- `iter/iter14_surgery_on_pretrain/plan_surgery_on_pretrain.md` — **THIS FILE** (Q&A + code plan)
- `src/utils/hf_finetuned_push.py` — DONE, used to push pretrain endpoint
- `src/m09c_surgery.py:259-276` — where `_load_init_state` will be patched
- `scripts/run_probe_train.sh` — where `pretrain_long` subcommand will be added
- `scripts/run_probe_eval.sh:143,178,188,404` — encoder resolver edit sites

---

## 📊 Empirical anchor (already proven by iter13)

| Metric | Frozen | Pretrain (5ep) | Δ vs Frozen |
|---|---:|---:|---|
| 🎯 `probe_top1` (motion-flow 16-class) | — | **0.808** | monotonic ↑ from 0.439 over 5 ep |
| 🌀 `motion_cos` (intra-vs-inter cosine) | 0.046 | **0.267** | **5.8×** — key paper signal |
| 🔮 `future_mse` | 0.5571 [0.5561, 0.5581] | **0.5544** [0.5531, 0.5557] | **Δ +0.0027, p=0.0** ✅ |
| 📉 `val_jepa_loss` | 0.473 | 0.458 | ↓ 3.2 % |
| 🧱 `block_drift_mean` | 0 | 0.0160 | healthy |
| 📏 `‖Δ‖/‖init‖` | 0 | **2.46 %** | non-collapsed |

🟢 **Half the strict ordering is locked**: `pretrain > frozen` with non-overlapping 95 % CI.

---

## ❓ Q1 — Preserving pretrain gains under surgery

### 🩺 Q1.1 — How to MONITOR loss of pretrain gains?

> 🛡️ Per-layer **CKA similarity** vs `θ^(pretrain)` (partially wired via `m09a_block_drift.png`); held-out **general-video probe** acc (K400 / SSv2 retention — **NOT yet tracked**); `val_jepa_loss` on a frozen pretrain-distribution slice; **gradient-norm spikes**; **weight-norm trajectory**; **"old probe" retention** (frozen probe trained on pretrain features, applied to surgery checkpoints — drop = forgetting).

### 🔧 Q1.2 — Other measures to PRESERVE pretrain gains (beyond drift control + LR decay)

| | Measure | What |
|---|---|---|
| 🔁 (c) | Replay | mix 5–10 % pretrain-distribution batches into surgery |
| 📐 (d) | EMA / weight averaging | high `τ ≥ 0.99` so teacher tracks slowly |
| 🎓 (e) | KL distillation | aux loss = KL(student_logits ‖ pretrain_logits) |
| 🪶 (f) | LoRA / adapter-only | freeze base, train low-rank delta |
| 🧮 (g) | EWC | `λ Σᵢ Fᵢ(θᵢ − θᵢ^pretrain)²` (Fisher-weighted) |
| ⚓ (h) | **L2 anchor loss** | `λ ‖θ − θ_pretrain‖²` (proposal Sec 10.6 — reused for pretrain→surgery) |
| 🛑 (i) | Early stopping | abort if pretrain-val loss rises |

### 🔥 Q1.3 — Handling Stage-1 catastrophic forgetting if backbone LR is too high

🚦 Cap backbone LR ≤ **1e-5** (vs predictor 1e-4) · 🌡️ short **100–500-step warmup** at each stage boundary · 📐 EMA `τ ≥ 0.99` · 🪜 layer-wise LR decay 0.7–0.9 · ⚓ anchor loss `λ ∈ [0.001, 0.01]` · ⚠️ early-abort surgery Stage-1 if `val_jepa` on pretrain-val rises **> 5 %**.

---

## ❓ Q2 — Reuse the pretrain checkpoint?

### ✅ Q2.1 — Is `m09a_pretrain` ckpt good to reuse for surgery init? **YES** ✅

> `probe_top1=0.808` (peaked at last step, no plateau) · `val_jepa ↓ 3.2 %` · `motion_cos ↑ 5.8×` · `block_drift` healthy at 1.6 % · `‖Δ‖/‖init‖=2.46 %` · `pretrain > frozen` on `future_mse` with non-overlapping 95 % CI (Δ +0.0027, p=0.0). Sound foundation for surgery.

### 🤗 Q2.2 — Push pretrain to HF? **DONE** ✅

Pushed via `src/utils/hf_finetuned_push.py` to `anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep`. Both `student_encoder.pt` (surgery init) AND `m09a_ckpt_best.pt` (probe_eval Stage 8 future_mse / `predictor` key) included.

---

## ❓ Q3 — Compute-matched long-pretrain ablation

### 🧮 Q3.1 — The control arm with numbers

| Arm | Composition | Total budget | Factor patching? |
|---|---|---|---|
| **A** — current pretrain | pretrain (5 ep · 1,010 steps · ~10 GPU-h) | 5 ep | ❌ |
| **B** — surgery on pretrain | pretrain (5 ep) ▶ surgery (5 ep) | **10 ep · ~20 GPU-h** | ✅ last 5 ep |
| **C** — long-pretrain control | pretrain (10 ep · 2,020 steps · ~20 GPU-h) | **10 ep · ~20 GPU-h** | ❌ |

🎯 **The proof**: if `B > C` with non-overlapping 95 % CI on `motion_cos` / `future_mse` / `probe_top1`, gain is **causal to factor patching**, not extra steps.

💰 Incremental cost: **one extra ~10 GPU-h training run** (long-pretrain control).

### 🤖 Q3.2 — Common in RL/RLHF papers? **YES**

Tülu 3 (Lambert et al., 2024) and *"Is DPO Superior to PPO?"* (Xu et al., 2024 OpenReview) both include compute-matched extended-SFT baselines. Canonical pattern: *"match total compute = SFT_steps + (DPO|PPO)_steps"*.

---

## 🛠️ T3 code plan — 7 file edits

> 🎯 ~120 LoC across 7 files. No architectural rewrites — surgical wiring only.

### 1️⃣ NEW · `configs/train/probe_pretrain_long.yaml` (~12 LoC)

```yaml
# iter14 — compute-matched control for surgery(5+5) vs long-pretrain(10)
# Identical to probe_pretrain.yaml except max_epochs.full doubled 5 → 10.
extends: probe_pretrain.yaml
optimization:
  max_epochs:
    sanity: 1
    poc: 2
    full: 10      # iter14 — was 5 in probe_pretrain.yaml
```

### 2️⃣ NEW · `configs/train/surgery_3stage_DI_iter14.yaml` (~10 LoC)

```yaml
# iter14 — surgery with 5-ep total budget (was 15 via base_optimization)
# Stage split: 2/2/1 across D_L → D_A → D_I.
extends: surgery_3stage_DI.yaml
optimization:
  max_epochs: { sanity: 1, poc: 1, full: 5 }   # was 15
drift_control:
  lambda_reg: 0.005       # iter14 — anchor to pretrain (was 0.0)
  anchor_to: pretrain     # NEW key — read by m09c (step 3); anchor target = loaded init ckpt
```

Mirror file: `surgery_2stage_noDI_iter14.yaml`.

### 3️⃣ EDIT · `src/m09c_surgery.py` (~25 LoC)

**Current** (lines 259-276): m09c always loads V-JEPA frozen ckpt from URL. **New**: optional `--init-from-ckpt` overrides.

```python
# argparse block (~line 1370):
parser.add_argument("--init-from-ckpt", default=None,
    help="iter14: load student weights from a prior training-run checkpoint "
         "(e.g. outputs/full/probe_pretrain/student_encoder.pt) INSTEAD of "
         "the frozen V-JEPA URL. Enables sequential SSL composition.")

# Replace lines 259-276 with this dispatcher:
def _load_init_state(ckpt_url_or_path, init_from_ckpt=None):
    """Resolve init state_dict from Meta V-JEPA URL OR a prior m09a/m09c export."""
    if init_from_ckpt is not None:
        path = Path(init_from_ckpt)
        assert path.exists(), f"--init-from-ckpt missing: {path}"
        print(f"  [iter14] Loading init from prior-run ckpt: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # Schema priority for prior-run exports (verified at HF endpoint 2026-05-08):
        #   1. m09a/m09c student_encoder.pt   → "student_state_dict" (588 keys for ViT-G)
        #   2. m09{a,c}_ckpt_best.pt           → "student" or nested "student_state_dict"
        #   3. fallback                        → flat state_dict
        for k in ("student_state_dict", "student", "state_dict"):
            if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    # Legacy frozen V-JEPA URL path (unchanged)
    print(f"Downloading pretrained weights: {ckpt_url_or_path}")
    ckpt = torch.hub.load_state_dict_from_url(ckpt_url_or_path, map_location="cpu", weights_only=False)
    if "target_encoder" in ckpt: return ckpt["target_encoder"]
    if "encoder" in ckpt: return ckpt["encoder"]
    return ckpt

# Anchor loss (~10 LoC inserted into train_step around existing JEPA loss):
if drift_cfg.get("anchor_to") == "pretrain" and theta_pretrain is not None:
    anchor_loss = sum(((p - p0) ** 2).sum()
                      for p, p0 in zip(student.parameters(), theta_pretrain))
    total_loss = jepa_loss + drift_cfg["lambda_reg"] * anchor_loss
```

🔑 The `student_state_dict` schema branch is **empirically verified** against the live HF endpoint (1.84 B params, 588 keys).

### 4️⃣ EDIT · `scripts/run_probe_train.sh` (~30 LoC)

Add `pretrain_long` subcommand + thread `--init-from-ckpt` through surgery:

```bash
# argparse case (~line 50): accept 4th subcommand
pretrain|pretrain_long|surgery_3stage_DI|surgery_noDI) ;;

# NEW dispatch branch (mirror of pretrain, different yaml + output dir):
pretrain_long)
    OUT_DIR="outputs/${mode_dir}/m09a_pretrain_long"
    TRAIN_CFG="configs/train/probe_pretrain_long.yaml"
    LAMBDA_REG=$(scripts/lib/yaml_extract.py "$TRAIN_CFG" drift_control.lambda_reg)
    # ... identical body to pretrain branch, substitute OUT_DIR + TRAIN_CFG ...
    ;;

# EDIT surgery_3stage_DI / surgery_noDI: auto-pass --init-from-ckpt iff present
PRETRAIN_CKPT="outputs/${mode_dir}/m09a_pretrain/student_encoder.pt"
INIT_FLAG=""
if [ -f "$PRETRAIN_CKPT" ]; then
    INIT_FLAG="--init-from-ckpt $PRETRAIN_CKPT"
    echo "  [iter14] surgery init from pretrain: $PRETRAIN_CKPT"
fi
# (append $INIT_FLAG to the python -u src/m09c_surgery.py invocation)

# Repoint surgery to iter14 yamls:
case "$SUBCMD" in
    surgery_3stage_DI) TRAIN_CFG="configs/train/surgery_3stage_DI_iter14.yaml" ;;
    surgery_noDI)      TRAIN_CFG="configs/train/surgery_2stage_noDI_iter14.yaml" ;;
esac
```

### 5️⃣ EDIT · `scripts/run_probe_eval.sh` (~15 LoC)

Add `vjepa_2_1_pretrain_long` to default `ENCODERS` + 3 resolver cases:

```bash
ENCODERS="${ENCODERS:-vjepa_2_1_frozen vjepa_2_1_pretrain vjepa_2_1_pretrain_long vjepa_2_1_surgical_3stage_DI vjepa_2_1_surgical_noDI}"

# encoder_ckpt_for() (line 178):
vjepa_2_1_pretrain_long) echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_long/student_encoder.pt" ;;
# encoder_predictor_ckpt_for() (line 188):
vjepa_2_1_pretrain_long) echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_long/m09a_ckpt_best.pt" ;;
# pretrain_cleanup_get_latest() (~line 404):
vjepa_2_1_pretrain_long) echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_long/m09a_ckpt_latest.pt" ;;
```

🟢 Existing pre-flight (lines 304-340) auto-drops missing-ckpt encoders, so partial state still evaluates cleanly.

### 6️⃣ EDIT · `src/probe_action.py --stage paired_delta` (~20 LoC)

Emit explicit Δ1 / Δ2 / Δ3 keys:

```python
ITER14_DELTAS = [
    ("delta_1_pretrain_vs_frozen",
     "vjepa_2_1_pretrain", "vjepa_2_1_frozen",
     "Δ1: continual SSL > frozen (proves domain adaptation works)"),
    ("delta_2_surgical_vs_pretrain",
     "vjepa_2_1_surgical_3stage_DI", "vjepa_2_1_pretrain",
     "Δ2: surgery > pretrain (proves factor patching adds value)"),
    ("delta_3_surgical_vs_pretrain_long",
     "vjepa_2_1_surgical_3stage_DI", "vjepa_2_1_pretrain_long",
     "Δ3: surgery > long-pretrain (CAUSAL — not extra steps)"),
]
out["iter14_paper_deltas"] = {}
for key, a, b, desc in ITER14_DELTAS:
    if a in encoder_metrics and b in encoder_metrics:
        d = compute_paired_bca(encoder_metrics[a], encoder_metrics[b], n_resamples=10000)
        d["interpretation"] = desc
        d["pass"] = (d["ci_lo"] > 0)
        out["iter14_paper_deltas"][key] = d
```

### 7️⃣ NEW · `iter/iter14_surgery_on_pretrain/runbook.md` (~50 LoC)

Just the canonical sequence — no logic:

```bash
# 1. Surgery (5+5) — runs FIRST; if Δ2 fails, abort long-pretrain
tmux new -s iter14
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL 2>&1 | tee logs/iter14_surgery_3stage_DI.log    # ~10h
# 2. Surgery_noDI (ablation)
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI --FULL      2>&1 | tee logs/iter14_surgery_noDI.log         # ~7h
# 3. Long-pretrain (control) — ONLY if surgery shows signal
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain_long --FULL     2>&1 | tee logs/iter14_pretrain_long.log        # ~20h
# 4. 5-encoder eval
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL                    2>&1 | tee logs/iter14_probe_eval.log           # ~4h
# 5. Inspect Δ1 / Δ2 / Δ3
jq '.iter14_paper_deltas' outputs/full/probe_action/probe_paired_delta.json
```

---

## ✅ Verification gates (per CLAUDE.md "VERIFY-FIRST")

| When | Gate | Action if fails |
|---|---|---|
| After step 1 | `outputs/full/m09c_surgery_3stage_DI/student_encoder.pt` exists; `probe_top1 ≥ 0.808` | check anchor_loss firing; bump `λ → 0.01` |
| After step 1 | `block_drift_mean < 0.05` in last surgery checkpoint | LR too high; cap backbone ≤ 1e-5 |
| After step 4 | `iter14_paper_deltas.delta_2.ci_lo > 0` | Δ2 fail = abort long-pretrain; report negative result |
| After step 4 | `iter14_paper_deltas.delta_3.ci_lo > 0` | Δ3 fail with Δ2 pass = weaker claim ("factor patching ≥ extra steps"); still publishable |

---

## 🧪 SANITY end-to-end smoke (run BEFORE committing FULL hours)

```bash
# ~25 min total on 24 GB or 96 GB
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain_long      --SANITY 2>&1 | tee logs/iter14_sanity_pretrain_long.log
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI  --SANITY 2>&1 | tee logs/iter14_sanity_surgery.log
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --sanity            2>&1 | tee logs/iter14_sanity_eval.log
```

---

## 🚦 Three approval gates before touching code

> 1. **Epoch budget** — 🅰️ **"5+5 vs 10"** (cheap, ~$33; recommended) or 🅱️ "5+15 vs 20" (~$57)?
> 2. **Anchor `λ`** — `0.005` (literature default) OR 3-point sweep `{0.001, 0.005, 0.01}` (3× surgery cost)?
> 3. **HF push of pretrain** — ✅ **DONE** (no decision needed).
>
> 💬 Reply *"go: 🅰️, λ=0.005"* (or your variant) and I'll execute T4 (7 file edits) + run the SANITY smoke.

---

## 📋 TODO state

| # | Task | Status |
|---|---|---|
| 185 | T1 — Q&A reformat | ✅ done |
| 186 | T2 — `plan_HIGH_LEVEL.md` rewrite | ✅ done |
| 187 | T3 — code plan (this file) | ✅ done · schema-validated against live HF |
| 188 | T4 — execute 7 file edits | 🟡 pending — needs gates 1+2 |
| 189 | T5 — SANITY smoke (~25 min) | 🔒 blocked by T4 |
| 190 | T6 — FULL training arms (~41 GPU-h) | 🔒 blocked by T5 |

---

## 📚 Sources

- 🎬 [V-JEPA 2: Self-Supervised Video Models (Assran et al., 2025)](https://arxiv.org/abs/2506.09985)
- 🎬 [V-JEPA 2 Meta AI page](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)
- 🔗 [Two-Stage Fine-Tuning Strategy survey](https://www.emergentmind.com/topics/two-stage-fine-tuning-strategy)
- 🔗 [Sequential Finetuning (SeqL)](https://www.emergentmind.com/topics/sequential-finetuning-seql)
- 🧠 [Continual Learning in Generative Models survey (2025)](https://www.arxiv.org/pdf/2506.13045v2)
- 🥊 [Is DPO Superior to PPO? (Xu et al., 2024)](https://openreview.net/pdf?id=6XH8R7YrSk) — compute-matched SFT baseline
- 🛠️ [DPO 2025 guide (Schmid)](https://www.philschmid.de/rl-with-llms-in-2025-dpo)
- 📖 [DPO tutorial (HF)](https://huggingface.co/blog/ariG23498/rlhf-to-dpo)
