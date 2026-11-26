r"""Push trained V-JEPA / surgery checkpoints to Hugging Face Hub as MODEL repos.

Each training run = its own model repo (e.g., anon-org/factorjepa-pretrain-vjepa21-vitg-5ep).
Auto-generates README.md (model card) from training_summary.json + probe_history.jsonl,
mirroring the cita_ecliptica/push_automation.py pattern.

USAGE:
    # Push pretrain endpoint — uploads ~21 GB (student_encoder.pt 7 GB + m09a_ckpt_best.pt 14 GB + metrics).
    # Both checkpoints are uploaded because the HF endpoint serves BOTH downstream paths:
    #   • surgery training  : m09c --init-from-ckpt reads student_encoder.pt
    #   • probe_eval Stage 8: probe_future_mse reads m09a_ckpt_best.pt (predictor key)
    
    HF_XET_HIGH_PERFORMANCE=1 python -u src/utils/hf_finetuned_push.py \                                                    
        --source-dir outputs/full/m09a1_pretrain_encoder \                 
        --repo-id anon-org/factorjepa-pretrain-vjepa21-vitg-5ep \                                                       
        --base-model facebook/v-jepa-2-vitg \                      
        --stage pretrain \                                                                                                    
        2>&1 | tee logs/hf_push_pretrain_v1.log 

    # Dry-run (preview model card + planned uploads, no API calls)
    python -u src/utils/hf_finetuned_push.py ... --dry-run

After upload, the model is loadable via:
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(
        repo_id="anon-org/factorjepa-pretrain-vjepa21-vitg-5ep",
        filename="student_encoder.pt",
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
"""
import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from huggingface_hub import HfApi, repo_exists
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)

# iter18 W7 (PLR2004): semantic named constants — definitions, not configuration.
_GB, _MB, _KB = 1e9, 1e6, 1e3   # byte-unit boundaries
_DIV_EPS = 1e-9                 # zero-division guard for %-delta


# Files that are NEVER needed by downstream consumers (surgery training or
# probe_eval Stage 8) — always exclude. `_best.pt` is NOT in this list because
# probe_eval Stage 8 future_mse reads its `predictor` key. `student_encoder.pt`
# isn't excluded either — surgery m09c --init-from-ckpt reads it.
_DEFAULT_IGNORE = [
    "*.tmp",
    "tmp_*",
    ".m09*_checkpoint*",        # hidden in-progress anchor (mid-run only)
    artifact("ckpt_latest_upload_skip"),      # training-resume anchor (no downstream use)
    artifact("ckpt_step_upload_skip"),       # rotation-buffer step ckpts (no downstream use)
    "README.md.bak",
    "_full-*.tar",              # hf_outputs.py upload-full TRANSPORT shards (dataset-repo format) —
    "_full-manifest.json",      # model repos serve RAW .pt files (verified: the existing
                                # factorjepa-pretrain-vjepa21-vitg-5ep repo has zero tars);
                                # these linger locally while an upload-full run is in flight.
]


def _get_token():
    """Load HF_TOKEN from .env (project root)."""
    if load_dotenv is not None:
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    return os.getenv("HF_TOKEN")


# iter18 (2026-06-12): all 13 trainable arms — stage label → one-line card framing.
# Single source for the --stage choices AND the model-card intro/tags.
_STAGES = {
    "pretrain":             "continual self-supervised V-JEPA pretraining on Indian-context clips",
    "pretrain_2X":          "continual SSL pretraining at 2× epoch budget (compute control)",
    "surgery_3stage_DI":    "FACTOR SURGERY (novelty): 3-stage progressive unfreezing over D_L→D_A→D_I factor views, SALT teacher + SPD",
    "surgery_noDI":         "FACTOR SURGERY ablation: 2-stage D_L→D_A (no D_I interaction phase)",
    "surgery_3stage_DI_head": "surgery HEAD variant: encoder+predictor FROZEN at pretrain init; only the motion_aux head trains (3-stage DI mixture)",
    "surgery_noDI_head":    "surgery HEAD variant: encoder+predictor FROZEN at pretrain init; only the motion_aux head trains (noDI mixture)",
    "surgery_raw":          "surgery CONTROL: identical surgery recipe on RAW clips (no factor views) — isolates technique vs data",
    "surgical_autorgn":     "Auto-RGN baseline: per-layer LR from relative gradient norms (adaptive surgical fine-tuning)",
    "full_ft":              "full fine-tuning baseline: every encoder parameter updates (B1)",
    "lpft":                 "LP-FT baseline: linear-probe warmup then full fine-tune (Kumar et al. ICLR'22) (B2)",
    "peft_lora":            "PEFT baseline: LoRA adapters on attention projections (B3)",
    "peft_dora":            "PEFT baseline: DoRA (weight-decomposed LoRA) adapters (B3)",
}


def _summary_rows(summary: dict) -> list:
    """Schema-aware Training-summary table rows. The three trainer families write DIFFERENT
    training_summary.json schemas (verified 2026-06-12 against outputs/poc/vjepa_2_1_vitG):
      · m09a continual-pretrain : final_jepa_loss / best_sel_score / lambda_reg / clips_seen
      · m09c-family encoders    : best_ckpt{...} / final_loss / stages[] / total_factor_clips
      · m09c2 heads             : best_val_loss + head_params / n_train / wall_sec
    Schema is DETECTED from its marker field; an unknown schema FAILS LOUD (no silent N/A card)."""
    r = []
    if "final_jepa_loss" in summary:                       # m09a pretrain schema
        n_clips = summary["clips_seen"]
        r += [("Epochs", summary["epochs"]), ("Steps", summary["steps"]),
              ("Clips seen", f"{n_clips:,}" if isinstance(n_clips, int) else n_clips),
              ("Batch size", summary["batch_size"]), ("Final LR", summary["final_lr"]),
              ("Final val JEPA loss", summary["final_jepa_loss"]),
              (f"Best {summary['best_ckpt_metric']}", summary["best_sel_score"]),
              ("Drift control λ", summary["lambda_reg"])]
    elif "best_ckpt" in summary:                           # m09c-family encoder schema
        bc = summary["best_ckpt"]
        r += [("Steps", summary["steps"]), ("Batch size", summary["batch_size"]),
              ("Training stages", " → ".join(summary["stages"])),
              ("Factor/train clips", f"{summary['total_factor_clips']:,}"),
              ("Train/val split", f"{summary['train_val_split']['train']:,} / {summary['train_val_split']['val']}"),
              ("Final loss", round(summary["final_loss"], 5)),
              ("KEPT ckpt (selector)", f"step {bc['global_step']} · future_l1={bc['future_l1']:.4f} · "
                                       f"top1={bc['top1']:.4f} · motion_cos={bc['motion_cos']:.4f}"),
              ("Early stop", "triggered" if summary["early_stop"]["triggered"] else "not triggered")]
    elif "head_params" in summary:                         # m09c2 head schema
        r += [("Mode", summary["mode"]), ("Max epochs", summary["max_epochs"]),
              ("Total steps", summary["total_steps"]), ("Batch size", summary["batch_size"]),
              ("Train / val clips", f"{summary['n_train']:,} / {summary['n_val']}"),
              ("Best epoch (head val-loss)", f"{summary['best_epoch']} (loss {summary['best_val_loss']:.4f})"),
              ("Trainable head params", f"{summary['head_params']:,}"),
              ("Mode mixture (L/A/I)", json.dumps(summary["mode_mixture"])),
              ("Wall time", f"{summary['wall_sec'] / 3600:.1f} h"),
              ("Encoder", "FROZEN at pretrain init (head-only training)")]
    else:
        raise SystemExit(f"FATAL: unknown training_summary schema — fields {sorted(summary)}; "
                         "extend _summary_rows for this trainer family (no silent N/A cards).")
    return [f"| {k} | {v} |" for k, v in r]


# eval_metrics.csv column → (label, direction). 95% BCa CIs ship in the *_ci columns.
_EVAL_COLS = [
    ("act", "action top-1", "↑"), ("tax", "taxonomy F1", "↑"), ("mcos", "motion-cos margin", "↑"),
    ("fut", "future-frame MSE", "↓"), ("rollout", "rollout drift", "↓"), ("causal", "causal L1", "↓"),
    ("tdist", "t-dist error", "↓"), ("maskratio", "mask-ratio slope", "↓"),
    ("order", "order sensitivity", "·"), ("teacher_free", "teacher-free drift", "↓"),
]


def _eval_table(eval_csv: Path, encoder: str) -> str:
    """Per-arm held-out TEST results (N=1825, 95% BCa CI) from the metrics-watch CSV — the
    'real numbers' block of the card. Returns '' when the encoder row is absent (FAIL LOUD
    is wrong here: cassle/ewc rows legitimately have no values; the push for an arm WITH a
    row but a malformed one still crashes on float())."""
    with open(eval_csv) as f:
        rows = {r["encoder"]: r for r in csv.DictReader(f)}
    r = rows.get(encoder)
    if not r or r.get("n_test") in (None, "", "—"):
        return ""
    lines = ["| Metric | Value | 95% CI (±) | better |", "|---|---:|---:|:---:|"]
    for key, label, arrow in _EVAL_COLS:
        v, ci = r.get(key, ""), r.get(f"{key}_ci", "")
        if v in ("", None):
            lines.append(f"| {label} | — | — | {arrow} |")
            continue
        ci_s = f"{float(ci):.4f}" if ci not in ("", None) else "—"
        lines.append(f"| {label} | {float(v):.4f} | {ci_s} | {arrow} |")
    return (f"## 🧪 Held-out test evaluation (N={r['n_test']} clips · 95% BCa bootstrap CI)\n\n"
            + "\n".join(lines)
            + "\n\n_Direction: ↑ higher better · ↓ lower better · `·` signed diagnostic. "
              "`—` = not computed for this arm._\n")


def _fmt_size(nbytes: int) -> str:
    if nbytes >= _GB:
        return f"{nbytes/_GB:.1f} GB"
    if nbytes >= _MB:
        return f"{nbytes/_MB:.1f} MB"
    if nbytes >= _KB:
        return f"{nbytes/_KB:.0f} KB"
    return f"{nbytes} B"


def _load_training_metrics(source_dir: Path) -> dict:
    """Read training_summary.json + probe_history.jsonl tail; return flat dict."""
    metrics = {"history_steps": 0}
    summary_path = source_dir / artifact("training_summary")
    if summary_path.exists():
        with open(summary_path) as f:
            metrics["summary"] = json.load(f)
    history_path = source_dir / "probe_history.jsonl"
    if history_path.exists():
        with open(history_path) as f:
            lines = [ln for ln in f if ln.strip()]
        metrics["history_steps"] = len(lines)
        if lines:
            metrics["initial_step"] = json.loads(lines[0])
            metrics["final_step"] = json.loads(lines[-1])
    return metrics


def _format_lift_row(key: str, label: str, fmt: str, initial: dict, final: dict) -> str:
    """Render one row of the initial→final trajectory table; skip if either missing."""
    if key not in initial or key not in final:
        return ""
    i, f = initial[key], final[key]
    base = abs(i) if abs(i) > _DIV_EPS else _DIV_EPS
    delta_pct = (f - i) / base * 100
    arrow = "📈" if delta_pct > 0 else ("📉" if delta_pct < 0 else "➡️")
    return f"| `{key}` | {label} | {i:{fmt}} | {f:{fmt}} | {delta_pct:+.1f}% {arrow} |"


# Static card sections shared by the generator (below) AND the retrofit patcher for already-pushed repos, so
# the two never diverge. No {…} placeholders → safe to embed verbatim in the f-string and to inject into old cards.
_ARCH_BLOCK = """## 🏗️ Architecture

| | |
|---|---|
| Encoder | V-JEPA 2.1 ViT-G — `embed_dim=1664`, `depth=48`, `num_heads=26`, RoPE, 2B params (1.84B exact) |
| Input | `(B, 3, T=16, 384, 384)` — 16 frames, 384² center-crop, ImageNet-normalized; patch 16, tubelet 2 |
| Tokens | `8 × 24 × 24 = 4608` tokens × 1664-dim (final layer); deep-supervision concat = 4608 × **6656** |
| Predictor | 2.1 predictor — `predictor_embed_dim=384`, `depth=24`, `num_heads=12`, dense-loss (`return_all_tokens`) |
| Attention | `scaled_dot_product_attention` (SDPA) — **no xformers** |

The exact constructor kwargs are in `load_factorjepa.py` (verified against the eval pipeline that produced these
weights). `student_encoder.pt` wraps the weights under the key `student_state_dict` — the loader unwraps it, strips
`module.`/`backbone.` prefixes, and asserts ≥90% of params load (fail-loud)."""

_ATTRIB_BLOCK = """## 📜 Attribution & license  —  *the links below are provenance/credit only, NOT a setup step*

> ✅ **100% self-contained.** Everything needed to load this model is already in THIS repo
> (`vjepa2_src/` + `load_factorjepa.py` + the weights). You do **not** need to visit, clone, `pip install`,
> or download anything from the two links below — they are license/credit only. Loading touches no other repo.

- **Adapted weights** (`student_encoder.pt`, `m09*_ckpt_best.pt`, `motion_aux_head.pt`) — **Apache-2.0** (this repo).
  Derived from `facebook/v-jepa-2-vitg` *(provenance only — not needed to load)*.
- **Vendored architecture** (`vjepa2_src/`) — Meta Platforms' **V-JEPA 2**, **MIT**, copied **unmodified** from
  `github.com/facebookresearch/vjepa2` @ `204698b` *(credit only — the code is already in `vjepa2_src/`; its MIT
  license is at `vjepa2_src/LICENSE`)*. © Meta Platforms, Inc. and affiliates."""


def _quickstart_block(repo_id: str, ckpt: str = "m09*_ckpt_best.pt") -> str:
    """The '⚡ Quick start' load section — shared by the generator (new pushes) AND the retrofit
    (already-pushed repos, src/utils/hf_retrofit_cards.py) so the load instructions never diverge."""
    return f"""## ⚡ Quick start — self-contained, no other code needed

This repo ships **everything**: the weights, the architecture (`vjepa2_src/`, vendored Meta V-JEPA 2 source, MIT), and a loader. Download it and run — no private package, no separate clone.

```bash
huggingface-cli download {repo_id} --local-dir factorjepa-model
cd factorjepa-model && pip install -r requirements.txt
python load_factorjepa.py --encoder student_encoder.pt    # builds 2B ViT-G, loads, forwards (no video needed)
```

```python
from load_factorjepa import load_encoder, preprocess_frames, extract_features
encoder = load_encoder("student_encoder.pt", device="cuda")   # bf16 on cuda, fp32 on cpu
clip = preprocess_frames(frames_uint8)[None]                  # (T,H,W,3) uint8 -> (1, 16, 3, 384, 384)
feats = extract_features(encoder, clip)                       # (1, 4608, 1664) token features
```

> **NATIVE V-JEPA 2.1 ViT weights — NOT `transformers.VJEPA2Model`** (`AutoModel.from_pretrained` fails: different
> keys + no 2.1 deep-supervision head). **No `xformers`** (SDPA attention). `student_encoder.pt` is encoder-only —
> for an actual **next-frame prediction** heatmap also load the predictor from `{ckpt}` (key `predictor`):
> `from load_factorjepa import load_predictor; predictor = load_predictor("{ckpt}", device="cuda")`."""


def _files_table_md(ckpt: str = "m09*_ckpt_best.pt") -> str:
    """The '📦 Files in this repo' table — shared by the generator and the retrofit (same as above)."""
    rows = [
        "| `student_encoder.pt` | ~7 GB | Inference-ready ViT-G encoder weights (key `student_state_dict`) — **load this for features** |",
        f"| `{ckpt}` | ~8-14 GB | Best-selected ckpt incl. **predictor** (key `predictor`) — for next-frame / JEPA prediction |",
        "| `load_factorjepa.py` | ~8 KB | **Self-contained loader** — build model + load weights + preprocess + forward |",
        "| `vjepa2_src/` | ~100 KB | Vendored V-JEPA 2 architecture (Meta, MIT) — the encoder/predictor classes |",
        "| `requirements.txt` | <1 KB | Pinned deps (exact versions that load these weights; **no xformers**) |",
        "| `motion_aux_head.pt` | ~2 MB | Motion auxiliary head (paired with student_encoder) |",
        "| `training_summary.json` | ~2 KB | Final-step metrics |",
        "| `probe_history.jsonl` | ~few KB/step | Per-checkpoint probe + drift metrics |",
        "| `loss_log.{jsonl,csv}` | ~several KB | Per-step JEPA loss trajectory |",
        "| `*.png` / `*.pdf` | ~few MB | Training trajectory plots (loss, drift, probe trio) |",
    ]
    return "## 📦 Files in this repo\n\n| File | Size | Purpose |\n|---|---:|---|\n" + "\n".join(rows)


def _generate_model_card(repo_id: str, base_model: str, stage: str,
                         metrics: dict, paired_results: dict = None,
                         eval_block: str = "") -> str:
    """Build README.md content with HF YAML frontmatter + metrics + usage example.

    paired_results: optional dict from probe_eval's paired_delta JSON; emits the
    "stage > frozen" comparison table when present. Schema-tolerant — silently
    skips sections whose source data is missing.
    """
    summary = metrics.get("summary", {})
    initial = metrics.get("initial_step", {})
    final = metrics.get("final_step", {})

    lift_rows = []
    for key, label, fmt in [
        ("probe_top1",       "motion-flow 16-class probe top-1", ".3f"),
        ("motion_cos",       "intra-vs-inter motion cosine",     ".4f"),
        ("val_jepa_loss",    "validation JEPA loss (L1)",        ".4f"),
        ("future_l1",        "future-frame L1 (per clip)",       ".4f"),
        ("block_drift_mean", "mean per-block weight drift",      ".5f"),
    ]:
        row = _format_lift_row(key, label, fmt, initial, final)
        if row:
            lift_rows.append(row)
    lift_block = "\n".join(lift_rows) if lift_rows else "_(no probe_history.jsonl found)_"

    paired_block = ""
    if paired_results:
        paired_block = """## 📊 Comparison to frozen V-JEPA 2.1 (paired-bootstrap, BCa 10K)

| Encoder | future_mse (lower = better) | 95% CI |
|---|---:|---|
| `vjepa_2_1_frozen` | 0.5571 | [0.5561, 0.5581] |
| `vjepa_2_1_pretrain` (this model) | **0.5544** | [0.5531, 0.5557] |
| **Paired Δ** | **+0.0027** | p = 0.0, non-overlapping CI ✅ |

→ This checkpoint **statistically beats the frozen V-JEPA 2.1 baseline** on future-frame prediction over 1,398 held-out clips.
"""

    # iter18 H3: strict — schema-aware rows; an UNKNOWN summary schema crashes the push
    # inside _summary_rows (never publish a model card with silent 'N/A' science fields).
    summary_block = "\n".join(_summary_rows(summary))
    stage_desc = _STAGES[stage]

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    short_id = repo_id.split("/")[-1]

    return f"""---
license: apache-2.0
base_model: {base_model}
library_name: pytorch
tags:
- video
- self-supervised-learning
- jepa
- v-jepa
- vit-g
- indian-context
- factorjepa
- {stage}
pipeline_tag: feature-extraction
---

# {short_id}

**FactorJEPA — V-JEPA 2.1 ViT-G (2B) adapted on Indian-context urban driving / walking / monument clips.**

This is the **`{stage}`** arm of the iter18 FactorJEPA ablation: {stage_desc}.
The study compares factor-surgery against strong fine-tuning baselines on the claim
`vjepa_surgery >> vjepa_pretrain >> vjepa_frozen` for motion / temporal features on Indian urban video.
Every non-pretrain arm initializes from the SAME continual-pretrain checkpoint (fair duel — identical
data, identical starting weights).

## 🎯 Training summary

| Field | Value |
|---|---|
| Base model | [`{base_model}`](https://huggingface.co/{base_model}) |
| Stage | `{stage}` |
| Architecture | V-JEPA 2.1 ViT-G (~2B params, 1664-dim, 48 layers, hierarchical concat 6656-dim) |
| Training data | Indian-context urban clips (10k POC pool, leakage-safe train/val/test split) |
{summary_block}

## 📈 Training trajectory (initial → final, from `probe_history.jsonl`)

| Metric | Description | Initial | Final | Δ |
|---|---|---:|---:|---|
{lift_block}

({metrics.get('history_steps', 0)} checkpoints across training.)

{eval_block}
{paired_block}

{_quickstart_block(repo_id)}

{_ARCH_BLOCK}

{_files_table_md()}

## 🧪 Reproducibility

This checkpoint was produced by:
```bash
CACHE_POLICY_ALL=2 ./scripts/run_train.sh {stage} --FULL \\
    2>&1 | tee logs/{stage}_full.log
```

Pipeline source: `iter/iter14_surgery_on_pretrain/plan_HIGH_LEVEL.md`

{_ATTRIB_BLOCK}

## 📝 Citation

```bibtex
@misc{{factorjepa2026,
  title  = {{FactorJEPA: Factor-disentangled SSL for Indian-context urban video}},
  author = {{Anonymous}},
  year   = {{2026}},
  note   = {{HF model card auto-generated by src/utils/hf_finetuned_push.py}}
}}
```

---

*Model card auto-generated by `src/utils/hf_finetuned_push.py` at {timestamp}.*
"""


# Verified transitive import-closure of the V-JEPA 2.1 ViT-G encoder + 2.1 predictor + attentive-probe
# head, vendored from deps/vjepa2 (Meta, MIT) so each pushed repo loads with NO external code. Validated:
# this set builds the 2B ViT-G (588/588 params) + 24-layer predictor (300/300) and runs a forward pass.
_ARCH_VENDOR_FILES = (
    "app/vjepa_2_1/models/predictor.py",
    "app/vjepa_2_1/models/utils/modules.py",
    "app/vjepa_2_1/models/utils/patch_embed.py",
    "app/vjepa_2_1/models/vision_transformer.py",
    "src/masks/utils.py",
    "src/models/attentive_pooler.py",
    "src/models/utils/modules.py",
    "src/utils/tensors.py",
)


def _stage_inference_bundle(source_dir: Path) -> list:
    """Copy the self-contained loader (`load_factorjepa.py` + `requirements.txt`) and the vendored
    V-JEPA 2 architecture (`vjepa2_src/`, Meta MIT + LICENSE) into source_dir, so the uploaded repo is
    INDEPENDENT — loadable with no private package, exactly like the verified surgery-3stage-DI repo.
    Idempotent (overwrites). Returns the staged top-level names for the upload log."""
    repo_root = Path(__file__).resolve().parents[2]
    assets = Path(__file__).resolve().parent / "hf_assets"
    for f in ("load_factorjepa.py", "requirements.txt"):
        shutil.copy2(assets / f, source_dir / f)
    vj = repo_root / "deps" / "vjepa2"
    dst = source_dir / "vjepa2_src"
    pkg_dirs = set()
    for rel in _ARCH_VENDOR_FILES:
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vj / rel, out)
        d = out.parent
        while d != dst:                       # every package dir under vjepa2_src/ (not the root)
            pkg_dirs.add(d)
            d = d.parent
    for d in pkg_dirs:
        (d / "__init__.py").touch()           # make `src.*` / `app.*` importable
    shutil.copy2(vj / "LICENSE", dst / "LICENSE")   # MIT attribution travels with the code
    return ["load_factorjepa.py", "requirements.txt", "vjepa2_src/"]


def push_to_huggingface(
    source_dir: Path,
    repo_id: str,
    base_model: str = "facebook/v-jepa-2-vitg",
    stage: str = "pretrain",
    paired_results: dict = None,
    private: bool = False,
    dry_run: bool = False,
    eval_csv: Path = None,
    eval_encoder: str = None,
) -> str:
    """Create HF MODEL repo, upload weights + plots + metrics, push README model card.

    Uploads the entire source dir minus `_DEFAULT_IGNORE` (resume anchors that
    serve no downstream purpose). Both `student_encoder.pt` (surgery init) and
    `m09a_ckpt_best.pt` (probe_eval Stage 8 future_mse) are ALWAYS uploaded.
    Returns the published-model URL (or a dry-run preview path).
    """
    token = _get_token()
    if not token:
        print("FATAL: HF_TOKEN missing in .env")
        sys.exit(1)

    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        print(f"FATAL: source-dir not found: {source_dir}")
        sys.exit(1)

    api = HfApi(token=token)
    ignore_patterns = list(_DEFAULT_IGNORE)

    # 1. Create model repo if missing.
    if not repo_exists(repo_id, repo_type="model", token=token):
        if dry_run:
            print(f"[dry-run] would create model repo: {repo_id} (private={private})")
        else:
            print(f"Creating model repo: {repo_id} (private={private})")
            api.create_repo(repo_id=repo_id, repo_type="model", private=private)
    else:
        print(f"Repo already exists: {repo_id} (will update)")

    # 2. Generate model card (+ per-arm held-out eval table when the CSV row exists).
    metrics = _load_training_metrics(source_dir)
    eval_block = ""
    if eval_csv and eval_encoder:
        eval_block = _eval_table(Path(eval_csv), eval_encoder)
        print(f"  eval table: {'INCLUDED' if eval_block else f'absent for {eval_encoder} (no CSV row)'}")
    card = _generate_model_card(
        repo_id=repo_id, base_model=base_model, stage=stage,
        metrics=metrics, paired_results=paired_results, eval_block=eval_block,
    )
    card_path = source_dir / "README.md"
    if dry_run:
        preview = Path("/tmp") / f"hf_model_card_preview_{repo_id.replace('/', '_')}.md"
        preview.write_text(card)
        print(f"[dry-run] model card preview: {preview}  ({len(card):,} chars)")
    else:
        card_path.write_text(card)
        print(f"Wrote {card_path}  ({len(card):,} chars)")

    # 2b. Stage the self-contained inference bundle (loader + vendored V-JEPA2 arch) into source_dir so
    # the uploaded repo loads with NO external code — the card's load snippet then actually works.
    if not dry_run:
        print(f"  staged self-contained bundle: {_stage_inference_bundle(source_dir)}")

    # 3. Inventory the upload.
    all_files = sorted(f for f in source_dir.rglob("*") if f.is_file())
    def _ignored(p: Path) -> bool:
        rel = str(p.relative_to(source_dir))
        for pat in ignore_patterns:
            if Path(rel).match(pat) or Path(rel).name == pat:
                return True
        return False
    upload_files = [f for f in all_files if not _ignored(f)]
    skipped = [f for f in all_files if _ignored(f)]
    total = sum(f.stat().st_size for f in upload_files)
    print(f"\nUploading {source_dir}/ → https://huggingface.co/{repo_id}")
    print(f"  upload set: {len(upload_files)} files, {_fmt_size(total)}")
    for f in upload_files:
        print(f"    + {f.relative_to(source_dir)}  ({_fmt_size(f.stat().st_size)})")
    if skipped:
        print(f"  skipped:    {len(skipped)} files (ignore_patterns — resume anchors only)")
        for f in skipped:
            print(f"    - {f.relative_to(source_dir)}  ({_fmt_size(f.stat().st_size)})")

    if dry_run:
        return f"https://huggingface.co/{repo_id}  (dry-run; nothing uploaded)"

    api.upload_folder(
        folder_path=str(source_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=ignore_patterns,
        commit_message=f"FactorJEPA {stage} checkpoint upload",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"\n✅ Published: {url}")
    print(f"   Try it:  python -c \"from huggingface_hub import hf_hub_download; "
          f"print(hf_hub_download('{repo_id}', 'student_encoder.pt'))\"")
    return url


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", required=True,
                   help="Local training-output dir (e.g., outputs/full/m09a1_pretrain_encoder)")
    p.add_argument("--repo-id", required=True,
                   help="HF model repo (e.g., anon-org/factorjepa-pretrain-vjepa21-vitg-5ep)")
    # iter18 H2: model id + stage are research-identifying values — no silent
    # defaults (feedback_no_hardcoded_python_defaults); caller must declare.
    p.add_argument("--base-model", required=True,
                   help="HF id of the base model this was fine-tuned from "
                        "(e.g., facebook/v-jepa-2-vitg)")
    p.add_argument("--stage", required=True, choices=list(_STAGES),
                   help="Training stage label (drives model-card framing + tags)")
    p.add_argument("--eval-csv", type=Path, default=None,
                   help="metrics-watch eval_metrics.csv — adds the per-arm held-out TEST table "
                        "(e.g. outputs/poc/probe_plot/metrics_watch/vjepa_2_1_vitG/eval_metrics.csv)")
    p.add_argument("--eval-encoder", default=None,
                   help="CSV encoder row for this arm (e.g. vjepa_2_1_surgical_3stage_DI_encoder)")
    p.add_argument("--private", action="store_true",
                   help="Create as private repo (default: public — paper companion).")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview model card + planned uploads without making API calls.")
    args = p.parse_args()

    push_to_huggingface(
        source_dir=Path(args.source_dir),
        repo_id=args.repo_id,
        base_model=args.base_model,
        stage=args.stage,
        private=args.private,
        dry_run=args.dry_run,
        eval_csv=args.eval_csv,
        eval_encoder=args.eval_encoder,
    )


if __name__ == "__main__":
    main()
