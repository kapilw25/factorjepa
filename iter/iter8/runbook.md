# 🚀 FactorJEPA Runbook
> 🏆 **High level Goal**: Surgery > ExPLoRA > Frozen on Prec@K, 115K clips.
> 🎯 **Mid level Goal**: Surgery > ExPLoRA > Frozen on Prec@K, 1K clips (val_1k) — the POC decision gate.
> 🧪 **Low level Goal (SANITY smoke test only)**: 3 stages pass end-to-end on 20 clips with non-NaN loss.

> 📊 **Tier ladder (2026-04-17 🕘 updated)**: SANITY (20 clips, ~60 s code check) → POC (1K clips / val_1k, ~10 h full pipeline) → FULL (115K clips, ~1.5 d on 96GB). The 100-dense-clip intermediate tier was validated on 2026-04-17 and then 🚚 retired — 3200 visits-per-clip at 100 scale produced overfitting-pressure Prec@K that wouldn't replicate at FULL, and the short run exposed two POC-config bugs (#60 max_epochs, #61 warmup_steps) which are now fixed. 1K × 20 epochs = 20 visits/clip is the cheapest scale where Prec@K signal is publishable.

> Run these commands on GPU. Verify each step before moving to the next.
> Architecture + decisions: `plan_TODO.md`. Error history: `errors_N_fixes.md`.

### 📊 Progress snapshot (updated 2026-04-19)

| Phase | Step | Status | Notes |
|---|---|---|---|
| Setup | GPU Setup | ✅ done | 11,444/11,458 files pulled, 112 GB local |
| A | m10 1K Grounded-SAM (GPU) | 🟢 RUNNING | re-run after 2026-04-19 `rm -rf outputs/poc/` accident; `logs/m10_1k_v1.log` at 145/1000 @ 7:56, ~3.3 s/clip, ETA ~47 min |
| B | m11 1K factor datasets (CPU) | ⏳ pending | ~10 min factor gen + ~30-60 min per-clip plots (~40-70 min total); `factor_manifest.json` lands at ~10 min — C can launch then, plots continue in background |
| C | 🎯 m09c POC surgery (GPU) | ⏳ **NEXT after B manifest lands** | ~2.7 h on full 96 GB GPU (no OOM contention); 33-point probe trajectory = early-abort signal |
| D | m05 1K frozen embed (GPU) | ⏳ pending — runs AFTER C | ~21 min measured (`m05_1k_frozen_v1.log:78`, 1000 clips in 1266 s); skip entirely if C's probe trajectory is flat/down, saves ~21 min |
| E | m05 1K surgical embed (GPU) | ⏳ pending — runs after D | ~21 min; sequential with D on GPU |
| F | 🏆 m06 Prec@K decision gate (GPU FAISS) | ⏳ pending — runs after E | ~5 min; reads D + E .npy, produces Surgery vs Frozen metric |
| G | ExPLoRA arm (m09b + m05 + m06) | 🔒 locked on F pass | conditional on Surgery > Frozen |
| FULL | 115K scale-up | 🔒 locked | conditional on POC win |

Legend: ✅ done · 🟢 running · ⏳ pending · ⏭️ skipped · 🎯 next target · 🏆 paper-decision gate · 🔒 blocked on upstream result · ❌ failed · 🚫 fatal stop

---

## 🛠️ GPU Setup (one-time, bare instance)

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
mkdir -p logs && ./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
# Step [9/9] pre-caches Grounding DINO weights (~1.8 GB) so m10 runs offline.
source venv_walkindia/bin/activate
chmod +x git_push.sh git_pull.sh
./git_push.sh --code-only "updating"  # sync code only (no HF upload, use on both Mac and GPU)
./git_pull.sh 2>&1 | tee logs/git_pull.log  # downloads outputs + data from HF
```

---

## 🧪 Pre-POC canary (run after any code change to m10 / utils/training / ch11_surgery.yaml)

### 🟢 m10 POC 1K canary (~15 s to confirm + ~47 min to finish) — RUNNING 2026-04-19, **IS Step A itself**
```bash
python -u src/m10_sam_segment.py --POC \
    --subset data/val_1k.json \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m10_1k_canary.log &
M10_PID=$!
sleep 15
grep -m1 "Clip limit:" logs/m10_1k_canary.log
```
- ✅ Got: `Clip limit: 1000` at line 20 of `logs/m10_1k_canary.log`. Process running at ~2.65 s/clip (faster than 6.13 s/clip projection — val_1k is lighter than dense100). ETA total ~47 min from kickoff.
- ⏭️ **This process IS Step A. Do NOT re-run the Step A block below** — outputs land in `outputs/poc/m10_sam_segment/` as expected. After `wait $M10_PID` completes, jump directly to Step B.

---

## 🟢 Step A: Grounded-SAM Segmentation — DINO + HF Sam3TrackerVideo — RUNNING (same process as canary step 2, do NOT re-invoke)

POC tier = 1000 clips from `data/val_1k.json`. Pipeline code validated on 100-dense-clip tier 2026-04-17 (6141 agents, 8712 interactions, quality_gate PASS at 6.13 s/clip); now scaled up.

> ⚠️ The canary block above is already running this exact command. Skip the `python -u src/m10_sam_segment.py` invocation in this section — re-running would `rm -rf outputs/poc/m10_sam_segment/` mid-flight and kill the in-progress work. Only run the **Verify** block after `wait $M10_PID` (from canary step 2) finishes.

```bash
rm -rf outputs/poc/m10_sam_segment/ outputs/poc/m11_factor_datasets/
python -u src/m10_sam_segment.py --POC \
    --subset data/val_1k.json \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m10_1k_v1.log
```

**Verify:**
```bash
cat outputs/poc/m10_sam_segment/summary.json | python3 -m json.tool
```

| Check | Expect (1000 clips val_1k, projected from 6.13 s/clip on 96GB Blackwell) |
|---|---|
| Wall time | ~6100 s = ~**102 min** (1000 × 6.13 s/clip; random-sample clips may run slightly faster than dense subset) |
| `n_total_agents` | ~40K-60K (extrapolated from 6141 on 100 dense; val_1k is random so lighter than dense subset) |
| `n_total_interactions` | ~60K-90K |
| `mean_agent_pixel_ratio` | 0.10-0.18 (lower than the 0.186 dense measurement — val_1k has some empty aerial/monument clips) |
| `mean_mask_confidence` | ≥ 0.85 |
| `clips_with_agents_pct` | ≥ 0.80 (val_1k random sample; some truly empty clips get correctly skipped) |
| `quality_gate` | `"PASS"` (4 checks) |
| `m10_overlay_verify/*.png` | 1000 saved; spot-check 20 — red masks on real agents, no FPs on wires/signage |

---

## ⏳ Step B: Factor Datasets (D_L + D_A + D_I with tight-bbox tubes) — CPU, ~10 min factor gen + ~30-60 min per-clip plots

> ⚠️ `rm -rf` first — stale partial m11 outputs (e.g. 2026-04-19 disk-full: 475 D_L / 25967 D_I / no `factor_manifest.json`) race with `verify_or_skip`.
> 🔀 Step C (surgery, GPU) launches as soon as `factor_manifest.json` lands (~10 min into B); m11's matplotlib plot loop keeps running on CPU in background, no GPU contention.

```bash
rm -rf outputs/poc/m11_factor_datasets/
python -u src/m11_factor_datasets.py --POC \
    --subset data/val_1k.json \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m11_1k_v1.log
```

**Verify:**

| Check | Expect (1000 clips val_1k, projected from 47 s / 100 dense with 32 CPU workers) |
|---|---|
| Wall time | **~40-70 min total**: ~10 min factor-gen (32-worker ProcessPool) + ~30-60 min per-clip 2×2 plot loop (single-threaded matplotlib, 1000 figures × 2-4 s each) |
| `factor_manifest.json` written | **at ~10 min mark** — D.2 unblocked here, plots continue in background |
| `D_L present` | 1000/1000 |
| `D_A present` | ≥ 800/1000 (same ~80% ratio as val_1k's `clips_with_agents_pct`) |
| `D_I present` | ≥ 750/1000 (slightly lower — some clips have agents but no interactions) |
| Total D_I tubes | ~60K-80K |
| Median tubes/clip | ~50 (broader distribution than dense 100 because val_1k is heterogeneous) |
| `m11_factor_samples.png` | D_L: agents blurred, layout sharp. D_A: agents bright, BG dimmed to 10% |
| `m11_per_Videoclip_verify/*.mp4` | 20 top videos, 2×2 H.264 grids, 960×540 @ 6fps |

```bash
python3 -c "
import json
m = json.load(open('outputs/poc/m11_factor_datasets/factor_manifest.json'))
tubes = [v['n_interaction_tubes'] for v in m.values()]
clips_with = sum(1 for t in tubes if t > 0)
print(f'D_I: {clips_with}/{len(tubes)} clips ({100*clips_with/len(tubes):.0f}%)  Total tubes: {sum(tubes)}')
"
```

---

## ⏳ Step D: V-JEPA 2.1 Frozen Embedding — 1000-clip val_1k — GPU, ~21 min measured

> ⚠️ `rm -rf` first — stale pre-#62 100-clip `embeddings_vjepa_2_1_frozen.npy` would be re-used silently by `verify_or_skip`.
> ⏸️ **POSTPONED to AFTER Step C (surgery).** Earlier plan ran this in parallel with B → would OOM when D.2 surgery allocates its optimizer state. Runs sequentially after C releases GPU. If C's probe trajectory is flat/down → skip D entirely, save ~21 min. Wall-time: `logs/m05_1k_frozen_v1.log:78` measured **1000 clips in 1266 s (0.79 clips/s)** on 96 GB Blackwell.

```bash
rm -rf outputs/poc/m05_vjepa_embed/
python -u src/m05_vjepa_embed.py --POC \
    --subset data/val_1k.json \
    --model-config configs/model/vjepa2_1.yaml \
    --encoder vjepa_2_1_frozen \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_1k_frozen_v1.log
```

**Verify:**
```bash
python3 -c "
import numpy as np
e = np.load('outputs/poc/m05_vjepa_embed/embeddings_vjepa_2_1_frozen.npy')
print(f'Shape: {e.shape}')   # expect (1000, 1664)
print(f'Range: [{e.min():.2f}, {e.max():.2f}]')
print(f'L2 norm mean: {np.linalg.norm(e, axis=1).mean():.2f}')"
```

| Check | Expect (1000 clips val_1k, projected from 4.23 s/clip on 96GB Blackwell, 100-dense measurement) |
|---|---|
| Wall time | **~21 min measured** (`logs/m05_1k_frozen_v1.log:78`: 1000 clips in 1266 s / 0.79 clips/s, 2026-04-19); earlier 70 min projection assumed 4.23 s/clip — actual is 1.27 s/clip on 96 GB after sizer ramps 8→44 |
| Per-clip rate | **~0.79 clips/s** (1.27 s/clip) steady-state; sizer ramps 8→44 over first ~600 clips |
| `embeddings_vjepa_2_1_frozen.npy` shape | `(1000, 1664)` |
| AdaptiveBatchSizer growth | 8 → 18+ (may hit max 44 at ~step 400+) |
| OOM events | 0 |
| Compile path | `torch.compile` ✅ (RoPE Q/K cast to V.dtype, #44/#59) |
| Model dtype | `bfloat16` (V-JEPA 2.1 native) |
| Branch routing | `frozen-native` (loads `target_encoder` from .pt, #42) |

---

## 🎯 Step C: Surgery Training — `src/m09c_surgery.py` 🏆 PRIMARY PATH (paper novelty)

> m09c = 3-stage progressive prefix unfreezing + factor datasets (D_L → D_A → D_I). No drift, no held-out val.
>
> **Why Step D (not E)**: immediate goal is `Surgery > Frozen` on Prec@K. D_L/D_A/D_I factors already built in Step B. Test this path FIRST — if it works, we have the paper result. ExPLoRA (Step E) is the comparison arm run AFTER Surgery is validated.

### 🎯 C — POC (1000 clips val_1k, ~2.7 h on 96GB, real training signal) — NEXT after B manifest lands

> **One-time yaml edit before running** (POC epoch count tuned for 1K scale so wall time stays under 3 h):
> ```bash
> # max_epochs.poc: 100 → 20 in configs/train/ch11_surgery.yaml
> # At 1K clips × 20 epochs × 32 BS = 620 total steps / 3 stages ≈ 207 per stage
> # vs. 100 clips × 100 epochs = 300 total steps / 3 = 99 per stage (old setting)
> sed -i 's/^  poc: 100.*# ~300 total/  poc: 20                       # 1K × 20 ep = 620 total steps (~2.7h)/' \
>     configs/train/ch11_surgery.yaml
> ```
> Warmup auto-scales via `warmup_pct: 0.20` (fix #60): 207-step stage → 41-step warmup → 166 steps at full LR per stage. Healthy ratio.

```bash
rm -rf outputs/poc/m09c_surgery/
python -u src/m09c_surgery.py --POC \
    --subset data/val_1k.json \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/ch11_surgery.yaml \
    --factor-dir outputs/poc/m11_factor_datasets/ \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09c_1k_surgery_v1.log
```

**Verify:**
```bash
ls -lh outputs/poc/m09c_surgery/student_encoder.pt outputs/poc/m09c_surgery/probe_trajectory.png outputs/poc/m09c_surgery/probe_history.jsonl
cat outputs/poc/m09c_surgery/training_summary.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['probe_trajectory_stats'])"
```
- `m09_train_loss.png`: 3 stage curves + vlines ~step 207/414, ≥0.03 loss drop post-warmup/stage, final Stage 3 ~0.35-0.42.
- `probe_trajectory_stats`: `bwt_prec_at_k > 0` + `monotonic: true` expected; `max_drop_prec_at_k > 0` = replay failed → inspect before D.4.
- Per-stage log: `[probe] stage=... N=1000 Prec@K=XX.XX±Y.YY mAP@K=... Cycle@K=...`

### ⏳ E — m05 re-embed on surgical student (~21 min on 96GB)

Apply surgery-trained V-JEPA 2.1 to the same 1000 val_1k clips so Prec@K is directly comparable with the frozen baseline (Step C). `rm -f` the surgical .npy only — keep frozen embeddings from Step C.

```bash
rm -f outputs/poc/m05_vjepa_embed/embeddings_vjepa_2_1_surgical.npy \
      outputs/poc/m05_vjepa_embed/embeddings_vjepa_2_1_surgical.paths.npy \
      outputs/poc/m05_vjepa_embed/.m05_checkpoint_vjepa_2_1_surgical.npz
python -u src/m05_vjepa_embed.py --POC \
    --subset data/val_1k.json \
    --model-config configs/model/vjepa2_1.yaml \
    --model outputs/poc/m09c_surgery/student_encoder.pt \
    --encoder vjepa_2_1_surgical \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_1k_surgical_v1.log
```

**Verify:**
```bash
python3 -c "
import numpy as np
e = np.load('outputs/poc/m05_vjepa_embed/embeddings_vjepa_2_1_surgical.npy')
print(f'Shape: {e.shape}')                             # expect (1000, 1664)
print(f'L2 norm mean: {np.linalg.norm(e, axis=1).mean():.2f}')"
```

### 🎯⏳ D.4 — m06 FAISS Prec@K metrics — 🏆 DECISION GATE (~2-5 min, FAISS-GPU)

Two m06 runs (frozen + surgical) to close the 🎯 decision gate. m06 takes `--encoder` (NOT `--local-data` — chain script error, see fix in errors log). Per-encoder JSON lands at `outputs/poc/m06_metrics_vjepa_2_1_*.json`.

```bash
# Frozen baseline — rm stale per-encoder JSON first
rm -f outputs/poc/m06_metrics_vjepa_2_1_frozen.json outputs/poc/knn_indices_vjepa_2_1_frozen.npy
python -u src/m06_faiss_metrics.py --POC \
    --subset data/val_1k.json \
    --encoder vjepa_2_1_frozen \
    --no-wandb \
    2>&1 | tee logs/m06_1k_frozen.log

# Surgical — the paper-arm result
rm -f outputs/poc/m06_metrics_vjepa_2_1_surgical.json outputs/poc/knn_indices_vjepa_2_1_surgical.npy
python -u src/m06_faiss_metrics.py --POC \
    --subset data/val_1k.json \
    --encoder vjepa_2_1_surgical \
    --no-wandb \
    2>&1 | tee logs/m06_1k_surgical.log
```

**Verify — decision gate:**
```bash
python3 -c "
import json
for name in ['frozen', 'surgical']:
    try:
        m = json.load(open(f'outputs/poc/m06_metrics_vjepa_2_1_{name}.json'))
        pk = m['easy']['precision_at_k']
        print(f'{name:12s}: Prec@K = {pk[\"mean\"]:.2f}% +/- {pk[\"ci\"][\"ci_half\"]:.2f}')
    except FileNotFoundError:
        print(f'{name:12s}: NOT YET RUN')
"
```

- **Surgery > Frozen (non-overlapping 95 % CIs)** → proceed to Step E (ExPLoRA comparison arm).
- **Surgery ≤ Frozen** → pause Step E; debug m10/m11 factor quality or revisit training config (warmup, epoch count — see #60).

---

## 🔒 Step E: ExPLoRA Training — `src/m09b_explora.py` (comparison arm, CONDITIONAL on D.4 pass)

> m09b = LoRA on blocks 2-47 + unfreeze blocks 0-1, no drift. Hardcoded ExPLoRA mode (no `--explora` flag).
>
> **Why Step E (not D)**: ExPLoRA is the adaptation-baseline comparator for `Surgery > ExPLoRA > Frozen`. Only valuable AFTER Step D Surgery passed — it completes the comparison triangle. If Surgery already ≤ Frozen, pause and debug factor quality before spending GPU on ExPLoRA.

### ⏳ E.1 — SANITY (20 clips, ~10 min, code smoke test before spending POC GPU time)

```bash
rm -rf outputs/sanity/m09b_explora/
python -u src/m09b_explora.py --SANITY \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/explora.yaml \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09b_sanity_explora.log
```

**Verify:** `ls -lh outputs/sanity/m09b_explora/student_encoder.pt` — exists, ~8 GB. Check log for `LoRA injection`, non-NaN `loss_jepa`, clean exit.

### 🔒 E.2 — POC (1000 clips val_1k, ~2 h, real training signal)

```bash
rm -rf outputs/poc/m09b_explora/
python -u src/m09b_explora.py --POC \
    --subset data/val_1k.json \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/explora.yaml \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09b_1k_explora.log
```

**Verify:** `ls -lh outputs/poc/m09b_explora/student_encoder.pt` — exists, ~8 GB. Check `loss_log.csv` for monotonically-decreasing JEPA loss (ExPLoRA has no stages, so it's a single clean curve).

### 🔒 E.3 — m05 re-embed on ExPLoRA student + m06 Prec@K

```bash
# m05 re-embed (~70 min) — rm stale ExPLoRA .npy only
rm -f outputs/poc/m05_vjepa_embed/embeddings_vjepa_2_1_explora.npy \
      outputs/poc/m05_vjepa_embed/embeddings_vjepa_2_1_explora.paths.npy \
      outputs/poc/m05_vjepa_embed/.m05_checkpoint_vjepa_2_1_explora.npz
python -u src/m05_vjepa_embed.py --POC \
    --subset data/val_1k.json \
    --model-config configs/model/vjepa2_1.yaml \
    --model outputs/poc/m09b_explora/student_encoder.pt \
    --encoder vjepa_2_1_explora \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_1k_explora.log

# m06 Prec@K for ExPLoRA
rm -f outputs/poc/m06_metrics_vjepa_2_1_explora.json outputs/poc/knn_indices_vjepa_2_1_explora.npy
python -u src/m06_faiss_metrics.py --POC \
    --subset data/val_1k.json \
    --encoder vjepa_2_1_explora \
    --no-wandb \
    2>&1 | tee logs/m06_1k_explora.log
```

---

**Subset = 1000 clips from `data/val_1k.json`** (random uniform, seed=99 per m00c). Same subset across Steps A–E so Prec@K is apples-to-apples across frozen / ExPLoRA / surgical arms.

> **SANITY vs POC rationale**: SANITY (D.1 / E.1) validates the code path (no crashes, non-NaN losses, checkpoint saves) on 20 clips in ~60 s — cheap smoke test. POC (D.2 / E.2) on 1000 val_1k clips is the actual training signal where loss curves and Prec@K deltas become interpretable. Run SANITY first, then POC only if SANITY passes.
>
> **🎯 Decision gate after D.2 + D.4 (Surgery POC + Prec@K)**: if Prec@K(surgery) > Prec@K(frozen) with non-overlapping 95 % CIs → paper result; proceed to E.1/E.2/E.3 for the full `Surgery > ExPLoRA > Frozen` comparison + ablations. If Surgery ≤ Frozen → **pause Step E** and debug factor quality (m10/m11 re-examination) FIRST — ExPLoRA offers no insight into a broken factoring signal.

---

## 🏁 Final 3-arm comparison (after all D + E steps complete)

```bash
python3 -c "
import json
for name in ['frozen', 'explora', 'surgical']:
    f = f'outputs/poc/m06_metrics_vjepa_2_1_{name}.json'
    try:
        m = json.load(open(f))
        pk = m['easy']['precision_at_k']
        print(f'{name:12s}: Prec@K = {pk[\"mean\"]:.1f}% +/- {pk[\"ci\"][\"ci_half\"]:.1f}')
    except FileNotFoundError:
        print(f'{name:12s}: NOT YET RUN')
"
```

**Win conditions**:
- Surgery > ExPLoRA > Frozen, all with non-overlapping 95 % CIs → paper headline, scale to 115K FULL.
- Surgery > Frozen, ExPLoRA ≈ Frozen → strong paper (surgery is the novelty, ExPLoRA failure is an ablation).
- Surgery ≈ ExPLoRA > Frozen → publishable but weakens the "surgery is special" claim.
- Surgery ≤ Frozen → pivot to temporal-interference-projection diagnostic paper (see `iter/utils/literarure_survey.md`).

---

## Shortcut: `./scripts/train_*.sh --POC` (single-command pipelines)

The individual steps above exist because you often want to re-run one step after a bug fix without re-running the whole pipeline. When everything is known-good, use the chained scripts:

```bash
./scripts/train_surgery.sh --POC 2>&1 | tee logs/surgery_1k_pipeline.log  # m10 → m11 → m09c → m05 → m06
./scripts/train_explora.sh --POC 2>&1 | tee logs/explora_1k_pipeline.log  # m05 frozen → m09b → m05 → m06
```

Both scripts currently hardcode `data/val_1k_local` + `data/val_1k.json` via their pre-flight checks (`train_surgery.sh:107-124`), so they match this runbook's POC tier exactly.
