# 🛠️ iter13 m06d — Coding Plan (P1 ✅ shipped · P2 + P3 next)

> ## 🎯 Paper goal —  Only Priority
> `vjepa_surgery` >> `vjepa_pretrain` >>  `vjepa_frozen` on the same probe protocol applied to our Indian eval_10k pool. 
> outperforms is represented by `>>` symbol
> Evalaution Metric: top@1 accuracy and other 2 metrics

---

## 📋 Status legend

| Emoji | Meaning |
|:-:|:--|
| ⬜ | Pending — not started |
| 🔄 | In progress |
| ✅ | Completed & verified |
| 🔥 | Critical / blocking gate |
| 📦 | Reuse existing infra (no edit) |
| ✏️ | Edit existing file |
| 🆕 | New file |

---

## ✅ P1 — SHIPPED 2026-05-03

`scripts/run_m06d_eval.sh` orchestrates 10 stages over `data/eval_10k_local/` (~9,951 Indian video clips):

| # | Module | Output | Time |
|:-:|:--|:--|:-:|
| 1️⃣ | `m06d_action_probe.py` | 4-layer attentive probe top-1 acc + 95 % BCa CI | GPU ~1.5 h |
| 2️⃣ | `m06d_motion_cos.py` | Per-clip motion-feature cosine similarity (intra–inter) | CPU ~5 min |
| 3️⃣ | `m06d_future_mse.py` | Future-frame latent prediction L1 (V-JEPA only) | GPU ~30 min |
| 4️⃣ | `m08d_plot_m06d.py` | 4-bar PNG/PDF comparison + per-LR loss/acc curves | CPU ~10 s |

SANITY validated `logs/run_src_m06d_sanity_v6.log` (50 clips/class, n_test = 22). Detailed P1 file specs preserved in **Appendix A** for paper writing.

---

## 📂 Dataset — `data/eval_10k_local`

| Field | Value |
|:--|:--|
| Source JSON | `data/eval_10k.json` (9,951 clip_keys, video-level uniform sample) |
| Local TARs | `data/eval_10k_local/subset-{00000..00009}.tar` |
| VLM tags | `data/eval_10k_local/tags.json` |

**Activity distribution** — used to derive 3-class (default) or 4-class (`--enable-monument-class`) splits via `m06d_action_probe.py:264-269` Stage 1:

| Class | Count | % | ≥ 50 / class |
|:--|:-:|:-:|:-:|
| 🚶 walking | 5,564 | 55.6 % | ✅ |
| 🚗 driving | 3,053 | 30.5 % | ✅ |
| 🚁 drone | 1,334 | 13.3 % | ✅ |
| 🏛️ monument | 49 | 0.5 % | 🟡 marginal |

Stratified 70/15/15 → ~6,966 train / ~1,492 val / ~1,492 test.

---

# 🥈 🥉 P2 + P3 — Coding plan

## 🎯 Goal

Extend the m06d eval pipeline to compare **4 encoders** end-to-end on the same Indian-context probe protocol:

| Slot | Encoder name | Source checkpoint |
|:--|:--|:--|
| 🥇 P1 control | `vjepa_2_1_frozen` | `checkpoints/vjepa2_1_vitG_384.pt` (Meta, on disk, 29 GB) |
| 🥇 P1 baseline | `dinov2` | `facebook/dinov2-with-registers-giant` (HF, online) |
| 🥈 **P2 (NEW)** | `vjepa_2_1_pretrain` | `outputs/<mode>/m06d_pretrain/student_encoder.pt` (built by `m09a_pretrain.py`) |
| 🥉 **P3 (NEW)** | `vjepa_2_1_surgical` | `outputs/<mode>/m06d_surgery/student_encoder.pt` (built by `m09c_surgery.py`) |

**Test plan**: SANITY on 24 GB RTX Pro 4000 → FULL on 96 GB RTX Pro 6000 Blackwell.

> 📌 **m09b ExPLoRA is dropped** — replaced by m09a continual SSL pretraining as the P2 baseline. m09b/explora.yaml move to `_archive/` (preserves git history; iter11 v3 D arm result remains citable).

---

## 🚦 Critical: data anchor for encoder training

P2 and P3 train the encoder on the **same TARs that `run_m06d_eval.sh:125-126` already reads** (`data/eval_10k_local/`), restricted to the **action_labels.json train-split keys** (~6,966 clips). The val and test splits stay held out from encoder training → no contamination, fair evaluation.

```text
data/eval_10k.json (9,951 clips)
        │
        ▼  m06d Stage 1 (run_labels_stage @ m06d_action_probe.py:254-279)
outputs/<mode>/m06d_action_probe/action_labels.json  (70/15/15 stratified)
        │
        ├── train keys (~6,966)  →  P2 m09a + P3 m09c encoder training
        ├── val   keys (~1,492)  →  m06d Stage 3 probe-head val + plateau early-stop
        └── test  keys (~1,492)  →  m06d Stage 4 paired-Δ gate (HELD OUT)
```

> ❌ **What we don't use**: `ultra_hard_3066_train.json` (2,452 clips, surgery's *legacy* training pool from `surgery_base.yaml`). That's a different distribution from m06d eval — using it would weaken P2-vs-P3 comparison.

---

## 📦 File-by-file change set (~140 LoC new code, ~95 % reuse)

### 1. 🆕 `configs/train/m06d_pretrain.yaml` (~85 LoC) — P2 config

Extends `base_optimization.yaml`. Three semantic deltas vs `ch10_pretrain.yaml`:

| Knob | ch10 (legacy) | m06d_pretrain (new) | Why |
|:--|:--|:--|:--|
| `optimization.lr` | 1.0e-6 | **1.0e-5** | 10× higher than ch10's failed value; 10× lower than Meta's continual peak |
| `optimization.max_epochs.full` | 15 | **5** | Per user spec — 5 epochs × ~6,966 clips ≈ 35K samples seen |
| `drift_control.type` | ewc + λ sweep | **l2_uniform** | Strong anchor, no Fisher computation overhead |
| `drift_control.lambda_reg` | null (sweep) | **1.0** | Single value matches `surgery_base.yaml:209` |
| `data.module` | m09a | **m09a** | Same trainer |
| `data.output_dir` | `outputs/full/ch10_pretrain` | **`outputs/<mode>/m06d_pretrain`** | Lives in m06d namespace |

Plus per-mode rows for SANITY 24 GB memory savers (`use_8bit_optim`, `gradient_checkpointing`, `paged_optim` = true under sanity; false under full) — mirrors `surgery_base.yaml:96-114`.

### 2. 🆕 `src/utils/m06d_train_subset.py` (~40 LoC, CPU)

Importable + CLI. Reads m06d Stage 1's `action_labels.json` and writes a flat subset JSON of train-split keys for m09a/m09c to consume:

```bash
python -u src/utils/m06d_train_subset.py \
    --action-labels outputs/full/m06d_action_probe/action_labels.json \
    --split train \
    --output data/eval_10k_train_split.json
# → {"clip_keys": [...6966 keys...], "n_clips": 6966, "split": "train", "source": "..."}
```

Mirrors `src/utils/eval_subset.py` (already used by `run_m06d_eval.sh:108-115` for SANITY subset gen).

### 3. 🆕 `scripts/run_m06d_train.sh` (~80 LoC) — P2/P3 trainer wrapper

Thin wrapper per CLAUDE.md "shell scripts are THIN wrappers". Two subcommands:

```bash
./scripts/run_m06d_train.sh pretrain --SANITY    # m09a continual SSL on eval_10k train-split
./scripts/run_m06d_train.sh surgery  --SANITY    # m09c factor surgery on same data
./scripts/run_m06d_train.sh pretrain --FULL      # 96 GB
```

**Sequence (pretrain)**:
1. Pre-flight: assert `outputs/<mode>/m06d_action_probe/action_labels.json` exists; if not, run m06d Stage 1 first
2. Generate `data/eval_10k_{train,val}_split.json` via `m06d_train_subset.py`
3. Invoke `m09a_pretrain.py "--$MODE" --train-config configs/train/m06d_pretrain.yaml --subset .../train_split.json --val-subset .../val_split.json --output-dir outputs/<mode>/m06d_pretrain --no-wandb`

**Sequence (surgery)**: same shape, invokes `m09c_surgery.py --train-config configs/train/surgery_3stage_DI.yaml --factor-dir outputs/full/m11_factor_datasets`. Pre-flight that m10/m11 factor datasets exist.

### 4. ✏️ `src/m09a_pretrain.py` (~15 LoC)

Two surgical edits:

**a)** Per-mode flatten in `merge_config_with_args()` (line ~131) — m06d_pretrain.yaml uses per-mode dicts for memory savers; mirror `m09c_surgery.py:130-139`:

```python
for k in ("use_8bit_optim", "gradient_checkpointing", "paged_optim"):
    v = cfg["optimization"].get(k)
    if isinstance(v, dict):
        cfg["optimization"][k] = v[mode_key]
```

**b)** Wire `enable_gradient_checkpointing` after `student.train()` in `train()` — mirror `m09c_surgery.py:418-420`:

```python
from utils.training import enable_gradient_checkpointing
if cfg["optimization"]["gradient_checkpointing"]:
    enable_gradient_checkpointing(student)
```

### 5. ✏️ `src/utils/frozen_features.py:54-67` (+12 LoC)

Add 2 entries to the `ENCODERS` registry — same `kind: vjepa`, `arch`, `crop`, `embed_dim` as the frozen entry; differ only in name:

```python
ENCODERS = {
    "vjepa_2_1_frozen":   {"kind": "vjepa", "arch": "vit_gigantic_xformers", "crop": 384, "embed_dim": 1664},
    "vjepa_2_1_pretrain": {"kind": "vjepa", "arch": "vit_gigantic_xformers", "crop": 384, "embed_dim": 1664},
    "vjepa_2_1_surgical": {"kind": "vjepa", "arch": "vit_gigantic_xformers", "crop": 384, "embed_dim": 1664},
    "dinov2":             {"kind": "dinov2", "model_id": "facebook/dinov2-with-registers-giant", "crop": 224, "embed_dim": 1536},
}
```

🚫 **No new loader function**. `m06d_action_probe.py:298-299` already does `if enc_kind == "vjepa": load_vjepa_2_1_frozen(args.encoder_ckpt, ...)`. The same loader handles all 3 V-JEPA ckpts (verified: `frozen_features.py:82-83` falls back through `target_encoder` → `encoder` → raw keys; `m09a_pretrain.py:872` `export_student_for_eval` writes `target_encoder`).

### 6. ✏️ `src/m06d_action_probe.py:435-482` — Stage 4 paired_delta N-way refactor

Currently hardcoded 2-way (V-JEPA vs DINOv2). Refactor to N-way matrix, ported from `m06d_future_mse.py:447-529` (already N-way and tested):

**New algorithm:**

```python
def run_paired_delta_stage(args, wb) -> None:
    # Auto-discover encoders by scanning subdirs that have all 3 expected files.
    enc_data = {}
    for enc_dir in sorted(args.output_root.iterdir()):
        if not enc_dir.is_dir():
            continue
        if all((enc_dir / f).exists() for f in
               ("test_predictions.npy", "test_clip_keys.npy", "test_metrics.json")):
            enc_data[enc_dir.name] = {
                "preds": np.load(enc_dir / "test_predictions.npy").astype(np.float32),
                "keys":  [str(k) for k in np.load(enc_dir / "test_clip_keys.npy", allow_pickle=True)],
                "agg":   json.loads((enc_dir / "test_metrics.json").read_text()),
            }
    available = sorted(enc_data.keys())
    if len(available) < 2:
        sys.exit(f"FATAL: need ≥ 2 encoders, found: {available}")

    by_encoder = {e: {"acc_pct": round(float(d["preds"].mean()) * 100, 4),
                      "n":       len(d["keys"]),
                      "top1_ci": d["agg"]["top1_ci"]} for e, d in enc_data.items()}

    pairwise_deltas = {}
    for i, a in enumerate(available):
        for b in available[i + 1:]:
            ka, kb = enc_data[a]["keys"], enc_data[b]["keys"]
            shared = sorted(set(ka) & set(kb))
            if not shared:
                continue
            ai = {k: i for i, k in enumerate(ka)}
            bi = {k: i for i, k in enumerate(kb)}
            a_arr = np.array([enc_data[a]["preds"][ai[k]] for k in shared], dtype=np.float32)
            b_arr = np.array([enc_data[b]["preds"][bi[k]] for k in shared], dtype=np.float32)
            delta = a_arr - b_arr
            bca = paired_bca(delta)
            pairwise_deltas[f"{a}_minus_{b}"] = {
                "n_shared":   int(len(shared)),
                "delta_pp":   round(float(delta.mean()) * 100, 4),
                "ci_lo_pp":   round(float(bca["ci_lo"]) * 100, 4),
                "ci_hi_pp":   round(float(bca["ci_hi"]) * 100, 4),
                "ci_half_pp": round(float(bca["ci_half"]) * 100, 4),
                "p_value":    float(bca["p_value_vs_zero"]),
                "gate_pass":  bool(bca["ci_lo"] > 0),
            }

    out = {"metric": "top1_accuracy", "by_encoder": by_encoder,
           "pairwise_deltas": pairwise_deltas}

    # Backward-compat shim — m08d_plot_m06d.py:213-302 reads these legacy top-level keys.
    if "vjepa_2_1_frozen" in by_encoder and "dinov2" in by_encoder:
        d = pairwise_deltas.get("vjepa_2_1_frozen_minus_dinov2", {})
        out.update({
            "n_clips_test":   d.get("n_shared"),
            "n_clips_vjepa":  by_encoder["vjepa_2_1_frozen"]["n"],
            "n_clips_dinov2": by_encoder["dinov2"]["n"],
            "n_clips_shared": d.get("n_shared"),
            "vjepa_acc_pct":  by_encoder["vjepa_2_1_frozen"]["acc_pct"],
            "dinov2_acc_pct": by_encoder["dinov2"]["acc_pct"],
            **{k: d.get(k) for k in ("delta_pp", "ci_lo_pp", "ci_hi_pp", "ci_half_pp", "p_value", "gate_pass")},
        })
    save_json_checkpoint(out, args.output_root / "m06d_paired_delta.json")
    log_metrics(wb, {"n_encoders_compared": len(available)})
    print(json.dumps(out, indent=2))
```

> ⚠️ **The legacy shim is mandatory** — `m08d_plot_m06d.py:213-302` reads `vjepa_acc_pct`, `dinov2_acc_pct`, `delta_pp`, `ci_half_pp`, `n_clips_shared` directly. Without the shim, plot Stage 10 crashes.

### 7. ✏️ `src/m06d_motion_cos.py:262-311` — Stage 7 paired_delta N-way refactor

Same pattern as #6. Reads `<encoder>/per_clip_motion_cos.npy` + `<encoder>/clip_keys_test.npy`. Output `m06d_motion_cos_paired.json` carries `by_encoder + pairwise_deltas` matrix + legacy `vjepa_score_mean / dinov2_score_mean / delta_mean` keys (per `m08d_plot_m06d.py:227, 301-302`).

### 8. ✏️ `src/m06d_future_mse.py:91-95` — extend `KNOWN_VARIANTS`

```python
KNOWN_VARIANTS = (
    "vjepa_2_1_frozen",
    "vjepa_2_1_pretrain",   # NEW (P2)
    "vjepa_2_1_surgical",   # NEW (P3)
)
```

`run_paired_per_variant_stage` (lines 447-529) already loops `KNOWN_VARIANTS` — no further change needed.

### 9. ✏️ `scripts/run_m06d_eval.sh` — extend ENCODERS + per-encoder ckpt resolver + Stage 8 V-JEPA loop

**a) Line 133** — extend default encoder list:

```bash
ENCODERS="${ENCODERS:-vjepa_2_1_frozen vjepa_2_1_pretrain vjepa_2_1_surgical dinov2}"
```

**b) Add encoder→checkpoint resolver functions** (after line 135). Two resolvers because Stages 2/3 only need the encoder, but Stage 8 future_mse also needs the predictor:

```bash
encoder_ckpt_for() {                          # encoder-only — for Stages 2/3
    case "$1" in
        vjepa_2_1_frozen)   echo "$ENCODER_CKPT" ;;                                       # checkpoints/vjepa2_1_vitG_384.pt
        vjepa_2_1_pretrain) echo "${DEFAULT_OUTPUT_PREFIX/sanity/full}/m06d_pretrain/student_encoder.pt" ;;
        vjepa_2_1_surgical) echo "${DEFAULT_OUTPUT_PREFIX/sanity/full}/m06d_surgery/student_encoder.pt" ;;
        *) echo "" ;;
    esac
}
encoder_predictor_ckpt_for() {                # encoder+predictor — for Stage 8 future_mse
    case "$1" in
        vjepa_2_1_frozen)   echo "$ENCODER_CKPT" ;;
        vjepa_2_1_pretrain) echo "${DEFAULT_OUTPUT_PREFIX/sanity/full}/m06d_pretrain/m09a_ckpt_best.pt" ;;
        vjepa_2_1_surgical) echo "${DEFAULT_OUTPUT_PREFIX/sanity/full}/m06d_surgery/m09c_ckpt_best.pt" ;;
        *) echo "" ;;
    esac
}
```

> 🔑 **Why two resolvers**: `student_encoder.pt` (`export_student_for_eval` at `training.py:939`) is **encoder-only**. Stage 8's `_load_predictor_2_1` (`m06d_future_mse.py:140`) requires a `"predictor"` key in the .pt — present only in `m09{a,c}_ckpt_best.pt` (full periodic checkpoints saved with `full=True` at `training.py:852`).

**c) Stage 2 features loop** (replace line 241 `EXTRA_CKPT="--encoder-ckpt $ENCODER_CKPT"`):

```bash
CKPT="$(encoder_ckpt_for "$ENC")"
EXTRA_CKPT=""
if [[ "$ENC" == vjepa* ]]; then
    [ -e "$CKPT" ] || { echo "FATAL: encoder ckpt missing for $ENC: $CKPT"; exit 3; }
    EXTRA_CKPT="--encoder-ckpt $CKPT"
fi
```

**d) Stage 8 (lines 352-365)** — currently hardcoded `--variant vjepa_2_1_frozen`. Loop V-JEPA variants:

```bash
if ! should_skip 8; then
    stamp "STAGE 8 · future_mse forward (GPU, V-JEPA variants only)"
    for ENC in $ENCODERS; do
        [[ "$ENC" == vjepa* ]] || continue
        CKPT="$(encoder_predictor_ckpt_for "$ENC")"
        [ -e "$CKPT" ] || { echo "FATAL: predictor-bearing ckpt missing for $ENC: $CKPT"; exit 3; }
        python -u src/m06d_future_mse.py "--$MODE" \
            --stage forward --variant "$ENC" --encoder-ckpt "$CKPT" \
            --action-probe-root "$OUTPUT_ACTION" \
            --local-data "$LOCAL_DATA" \
            --output-root "$OUTPUT_MSE" \
            --num-frames "$NUM_FRAMES" --cache-policy "$P_MSE" \
            2>&1 | tee "logs/m06d_future_mse_forward_${ENC}.log"
    done
fi
```

**e) Pre-flight P2/P3 trainer outputs** (after line 222, before Stage 1) — auto-drop encoders whose checkpoints aren't yet trained:

```bash
if [ "$MODE" != "SANITY" ]; then
    for ENC in $ENCODERS; do
        case "$ENC" in
            vjepa_2_1_pretrain|vjepa_2_1_surgical)
                CKPT="$(encoder_ckpt_for "$ENC")"
                if [ ! -e "$CKPT" ]; then
                    echo "  WARN: $ENC ckpt $CKPT not found — train via:"
                    case "$ENC" in
                        vjepa_2_1_pretrain) echo "    ./scripts/run_m06d_train.sh pretrain --$MODE" ;;
                        vjepa_2_1_surgical) echo "    ./scripts/run_m06d_train.sh surgery  --$MODE" ;;
                    esac
                    echo "  → removing $ENC from this run's ENCODERS"
                    ENCODERS="${ENCODERS//$ENC/}"
                fi ;;
        esac
    done
fi
```

### 10. 📦 No-change reuse

| File | Why no change |
|:--|:--|
| `src/m09a_pretrain.py` (trainer trunk: `build_model`, `train`, `_train_step_grad_accum`, ckpt I/O) | Already supports continual SSL when `cfg.meta.load_checkpoint=True`; m06d_pretrain.yaml just retunes hyperparameters |
| `src/m09c_surgery.py` (P3 trainer) | Already shipped + validated for `surgery_3stage_DI` |
| `src/utils/training.py` (compute_drift_loss, _train_step_grad_accum, etc.) | Technique-agnostic primitives — no edits needed |
| `src/utils/frozen_features.py:72-101` `load_vjepa_2_1_frozen` | Path-driven loader handles all 3 V-JEPA ckpts via key fallback |
| `src/m06d_future_mse.py:447-529` `run_paired_per_variant_stage` | Already N-way; reused as algorithmic template for #6 + #7 |
| `src/m08d_plot_m06d.py` | Read-only consumer; backward-compat shim in #6/#7 keeps it working |
| `configs/train/{surgery_3stage_DI,surgery_base,base_optimization}.yaml` | P3 config chain unchanged |

---

## 🧪 Test sequence — 24 GB SANITY → 96 GB FULL

### 🅰️ Phase A — SANITY on RTX Pro 4000 (24 GB)

```bash
# A.1 — Generate action_labels.json + train/val/test splits (~1 min CPU)
SKIP_STAGES="2,3,4,5,6,7,8,9,10" CACHE_POLICY_ALL=2 \
    ./scripts/run_m06d_eval.sh --sanity 2>&1 | tee logs/run_m06d_sanity_stage1.log

# A.2 — Train P2 on the SANITY train-split (~70 clips, 1 epoch, ~3 min GPU)
./scripts/run_m06d_train.sh pretrain --SANITY 2>&1 | tee logs/m06d_pretrain_sanity.log
# → Expects VRAM peak <22 GB (8-bit optim + grad-ckpt)
# → Writes outputs/sanity/m06d_pretrain/{m09a_ckpt_best.pt, student_encoder.pt}

# A.3 — Train P3 on the SANITY train-split (~70 clips, 1 epoch/stage, ~10 min GPU)
./scripts/run_m06d_train.sh surgery --SANITY 2>&1 | tee logs/m06d_surgery_sanity.log

# A.4 — Run m06d Stages 2-10 with all 4 encoders (~10 min GPU)
ENCODERS="vjepa_2_1_frozen vjepa_2_1_pretrain vjepa_2_1_surgical dinov2" \
SKIP_STAGES="1" CACHE_POLICY_ALL=2 \
    ./scripts/run_m06d_eval.sh --sanity 2>&1 | tee logs/run_m06d_sanity_4enc.log

# A.5 — Verify outputs
jq '.pairwise_deltas | keys' outputs/sanity/m06d_action_probe/m06d_paired_delta.json
# → 6 keys (C(4,2) = 6 pairwise comparisons)
ls outputs/sanity/m08d_plot_m06d/
# → m06d_action_probe_loss/acc.png, m06d_encoder_comparison.png (4-bar)
```

**SANITY pass criteria**: all stages exit 0; `pairwise_deltas` has 6 entries; m08d plots render without crash; Stage 8 future_mse runs for all 3 V-JEPA variants without predictor-key error. Numbers are NOT meaningful (n_test ≈ 22).

### 🅱️ Phase B — FULL on RTX Pro 6000 Blackwell (96 GB)

```bash
# B.1 — Generate splits (~1 min CPU)
SKIP_STAGES="2,3,4,5,6,7,8,9,10" CACHE_POLICY_ALL=2 \
    ./scripts/run_m06d_eval.sh 2>&1 | tee logs/run_m06d_full_stage1.log

# B.2 — Train P2 (~3 GPU-h on 96 GB)
tmux new -s p2_full
./scripts/run_m06d_train.sh pretrain --FULL 2>&1 | tee logs/m06d_pretrain_full.log

# B.3 — Train P3 (~6-8 GPU-h on 96 GB)
tmux new -s p3_full
./scripts/run_m06d_train.sh surgery --FULL 2>&1 | tee logs/m06d_surgery_full.log

# B.4 — Run m06d Stages 2-10 with 4 encoders (~3 GPU-h)
tmux new -s m06d_full
ENCODERS="vjepa_2_1_frozen vjepa_2_1_pretrain vjepa_2_1_surgical dinov2" \
SKIP_STAGES="1" CACHE_POLICY_ALL=1 \
    ./scripts/run_m06d_eval.sh 2>&1 | tee logs/run_m06d_full_4enc.log

# B.5 — Read the gates
jq '.pairwise_deltas' outputs/full/m06d_action_probe/m06d_paired_delta.json
# → 🥇 P1: vjepa_2_1_frozen_minus_dinov2          (target Δ ≥ +20 pp, gate_pass=true)
# → 🥈 P2: vjepa_2_1_pretrain_minus_vjepa_2_1_frozen   (target Δ > 0)
# → 🥉 P3: vjepa_2_1_surgical_minus_vjepa_2_1_pretrain (target Δ > 0)
```

---

## ⚠️ Risk register (highest priority first)

| # | Risk | Mitigation |
|:-:|:--|:--|
| **R8** 🔥 | `student_encoder.pt` is encoder-only (`training.py:939`) → Stage 8 future_mse FATALs on missing `predictor` key for pretrain/surgical | **Resolved by #9b**: split per-stage ckpt resolver — Stages 2/3 read `student_encoder.pt`; Stage 8 reads `m09{a,c}_ckpt_best.pt` (full ckpt with predictor) |
| **R7** 🔥 | `m08d_plot_m06d.py:213-302` reads legacy keys (`vjepa_acc_pct`, etc.) → crashes on N-way schema | **Resolved by #6 + #7**: backward-compat shim emits both `pairwise_deltas` matrix AND legacy keys |
| **R2** | Drift `lambda_reg=1.0` calibration unknown — could dominate loss (no learning) or be noise (no anchor) | Watch SANITY's `loss_log.csv:loss_drift` first 100 steps. If > 10× jepa_loss → reduce to 0.01. If < 0.01× → bump to 100 |
| **R3** | OOM at 24 GB SANITY despite memory savers (28 trainable blocks for `freeze_below=20`) | Fallback ladder: `freeze_below: 20 → 32` (top 16 only); reduce SANITY train clips; `--batch-size 32 → 16`; AdaptiveBatchSizer auto-shrinks micro-batch |
| **R10** | `bitsandbytes` dep may be missing on 24 GB image | Pre-flight `python -c "import bitsandbytes"` in `run_m06d_train.sh` |
| **R-data** | Train/test contamination — eval_10k.json shared across encoder pretrain + probe eval | **Resolved by §"Critical data anchor"**: P2/P3 train ONLY on action_labels.json train-split keys; test split held out |
| **R-deprecate** | m09b_explora.py + explora.yaml + 4 stale `surgery_*.yaml` configs become deprecated | Move to `configs/train/_archive/` and `src/_archive/` (preserves git history) — separate cleanup task |

---

## ✅ Verification (end-to-end after implementation)

1. **3-check gate** on every edited .py: `py_compile`, `ast.parse`, `ruff check --select F,E9` — auto-runs via `post-edit-lint.sh` hook
2. **Preflight skill** on `m06d_action_probe.py`, `m06d_motion_cos.py`, `m06d_future_mse.py`, `run_m06d_eval.sh`, `m09a_pretrain.py`: `/preflight @<file>` for each
3. **REPL shim test** — synthesize a 4-encoder paired_delta JSON, load via `m08d_plot_m06d._load_paired_delta()` + call `plot_encoder_comparison()` → must produce 4-bar PNG without crash
4. **SANITY end-to-end** (Phase A above) — `logs/run_m06d_sanity_4enc.log` exits 0, all 6 pairwise deltas computed, m08d plots render
5. **FULL run** — Phase B sequence, only after SANITY passes

---

## 📝 Files touched

| File | Change | LoC |
|:--|:--|:-:|
| `configs/train/m06d_pretrain.yaml` | 🆕 NEW | ~85 |
| `src/utils/m06d_train_subset.py` | 🆕 NEW | ~40 |
| `scripts/run_m06d_train.sh` | 🆕 NEW | ~80 |
| `src/m09a_pretrain.py` | ✏️ per-mode flatten + grad-ckpt wiring | +15 |
| `src/utils/frozen_features.py:54-67` | ✏️ ENCODERS +2 entries | +12 |
| `src/m06d_action_probe.py:435-482` | ✏️ paired_delta N-way + legacy shim | +60 / -45 |
| `src/m06d_motion_cos.py:262-311` | ✏️ paired_delta N-way + legacy shim | +55 / -40 |
| `src/m06d_future_mse.py:91-95` | ✏️ KNOWN_VARIANTS +1 entry | +1 |
| `scripts/run_m06d_eval.sh` | ✏️ ENCODERS + ckpt resolvers + Stage 8 loop + pre-flight | +35 |
| **Total NEW code** | | **~140 LoC** |
| **Total reuse** | | **~95 %** |

---

## 📚 Appendix A — P1 file specs (preserved for paper writing)

P1 ships 5 files (3 modules + 2 utils) + 1 orchestrator + 1 plotter. Detailed function-level specs (pre-implementation drafts, ~1,000 LoC of documentation) are preserved in git history at commit `76211e3` (and earlier) for reference. The runtime behavior is the source of truth — all specs are reflected in the live source files:

| Spec section | Live file | Key entry points |
|:--|:--|:--|
| File 1 — `action_labels.py` | `src/utils/action_labels.py` | `parse_action_from_clip_key`, `load_subset_with_labels`, `stratified_split`, `write/load_action_labels_json` |
| File 2 — `vjepa2_imports.py` | `src/utils/vjepa2_imports.py` | `get_attentive_classifier()` |
| File 3 — `m06d_action_probe.py` 🔥 | `src/m06d_action_probe.py` | Stages: `labels`, `features`, `train`, `select_best_lr`, `paired_delta` |
| File 4 — `m06d_motion_cos.py` | `src/m06d_motion_cos.py` | Stages: `features`, `cosine`, `paired_delta` |
| File 5 — `m06d_future_mse.py` | `src/m06d_future_mse.py` | Stages: `forward`, `paired_per_variant` |
| Plotter (added post-spec) | `src/m08d_plot_m06d.py` | `plot_loss_curves`, `plot_acc_curves`, `plot_encoder_comparison` |
| Orchestrator (post-spec) | `scripts/run_m06d_eval.sh` | 10 stages, mode-gated SANITY/POC/FULL, upfront cache-policy gather |

---

## 🔗 Cross-references

- 🎯 Paper goals: `iter/iter13_motion_probe_eval/plan_training.md`
- 📊 Analysis (Q2 measurements + decision matrices): `iter/iter13_motion_probe_eval/analysis.md`
- 📒 Coding contract: `src/CLAUDE.md`
- 🐛 Error log: `iter/iter13_motion_probe_eval/errors_N_fixes.md`
- 🛡️ Preflight CPU-side guards: `.claude/skills/preflight/SKILL.md`
- 🧪 P1 SANITY validation log: `logs/run_src_m06d_sanity_v6.log`
