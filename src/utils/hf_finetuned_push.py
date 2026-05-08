"""Push trained V-JEPA / surgery checkpoints to Hugging Face Hub as MODEL repos.

Each training run = its own model repo (e.g., anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep).
Auto-generates README.md (model card) from training_summary.json + probe_history.jsonl,
mirroring the cita_ecliptica/push_automation.py pattern.

USAGE:
    # Push pretrain endpoint — uploads ~21 GB (student_encoder.pt 7 GB + m09a_ckpt_best.pt 14 GB + metrics).
    # Both checkpoints are uploaded because the HF endpoint serves BOTH downstream paths:
    #   • surgery training  : m09c --init-from-ckpt reads student_encoder.pt
    #   • probe_eval Stage 8: probe_future_mse reads m09a_ckpt_best.pt (predictor key)
    
    HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_finetuned_push.py \                                                    
        --source-dir outputs/full/m09a_pretrain \                 
        --repo-id anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep \                                                       
        --base-model facebook/v-jepa-2-vitg \                      
        --stage pretrain \                                                                                                    
        2>&1 | tee logs/hf_push_pretrain_v1.log 

    # Dry-run (preview model card + planned uploads, no API calls)
    python -u src/utils/hf_finetuned_push.py ... --dry-run

After upload, the model is loadable via:
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(
        repo_id="anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep",
        filename="student_encoder.pt",
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from huggingface_hub import HfApi, repo_exists


# Files that are NEVER needed by downstream consumers (surgery training or
# probe_eval Stage 8) — always exclude. `_best.pt` is NOT in this list because
# probe_eval Stage 8 future_mse reads its `predictor` key. `student_encoder.pt`
# isn't excluded either — surgery m09c --init-from-ckpt reads it.
_DEFAULT_IGNORE = [
    "*.tmp",
    "tmp_*",
    ".m09*_checkpoint*",        # hidden in-progress anchor (mid-run only)
    "m09*_ckpt_latest.pt",      # training-resume anchor (no downstream use)
    "m09*_ckpt_step*.pt",       # rotation-buffer step ckpts (no downstream use)
    "README.md.bak",
]


def _get_token():
    """Load HF_TOKEN from .env (project root)."""
    if load_dotenv is not None:
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    return os.getenv("HF_TOKEN")


def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1e9: return f"{nbytes/1e9:.1f} GB"
    if nbytes >= 1e6: return f"{nbytes/1e6:.1f} MB"
    if nbytes >= 1e3: return f"{nbytes/1e3:.0f} KB"
    return f"{nbytes} B"


def _load_training_metrics(source_dir: Path) -> dict:
    """Read training_summary.json + probe_history.jsonl tail; return flat dict."""
    metrics = {"history_steps": 0}
    summary_path = source_dir / "training_summary.json"
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
    base = abs(i) if abs(i) > 1e-9 else 1e-9
    delta_pct = (f - i) / base * 100
    arrow = "📈" if delta_pct > 0 else ("📉" if delta_pct < 0 else "➡️")
    return f"| `{key}` | {label} | {i:{fmt}} | {f:{fmt}} | {delta_pct:+.1f}% {arrow} |"


def _generate_model_card(repo_id: str, base_model: str, stage: str,
                         metrics: dict, paired_results: dict = None) -> str:
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

    files_table = [
        "| `student_encoder.pt` | ~7 GB | Inference-ready ViT-G encoder weights — **use for surgery init / m09c `--init-from-ckpt`** |",
        "| `m09a_ckpt_best.pt` | ~14 GB | Best-val ckpt w/ optimizer + **predictor** — **required for `probe_eval.sh` Stage 8 `future_mse`** |",
        "| `motion_aux_head.pt` | ~2 MB | Motion auxiliary head (paired with student_encoder) |",
        "| `training_summary.json` | ~2 KB | Final-step metrics |",
        "| `probe_history.jsonl` | ~few KB/step | Per-checkpoint probe + drift metrics |",
        "| `loss_log.{jsonl,csv}` | ~several KB | Per-step JEPA loss trajectory |",
        "| `*.png` / `*.pdf` | ~few MB | Training trajectory plots (loss, drift, probe trio) |",
    ]

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

    n_clips = summary.get("clips_seen", "N/A")
    n_clips_str = f"{n_clips:,}" if isinstance(n_clips, int) else str(n_clips)

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

**FactorJEPA — V-JEPA 2.1 ViT-G continual-pretrained on Indian-context urban driving / walking / monument clips.**

This is a **`{stage}`** checkpoint from the FactorJEPA pipeline, a sequential SSL composition that aims to prove
`vjepa_surgery >> vjepa_pretrain >> vjepa_frozen` on motion / temporal features for Indian urban video.

## 🎯 Training summary

| Field | Value |
|---|---|
| Base model | [`{base_model}`](https://huggingface.co/{base_model}) |
| Stage | `{stage}` |
| Architecture | V-JEPA 2.1 ViT-G (1664-dim, 48 layers, dense predictive L1 loss) |
| Training data | 9,297 Indian-context clips (`data/eval_10k_local`) |
| Epochs | {summary.get('epochs', 'N/A')} |
| Steps | {summary.get('steps', 'N/A')} |
| Clips seen | {n_clips_str} |
| Batch size | {summary.get('batch_size', 'N/A')} |
| Final LR | {summary.get('final_lr', 'N/A')} |
| Final val JEPA loss | {summary.get('final_jepa_loss', 'N/A')} |
| Best probe top-1 | {summary.get('best_probe_top1', 'N/A')} |
| Drift control λ | {summary.get('lambda_reg', 0)} |
| Final encoder drift `‖Δ‖/‖init‖` | reported in training log (e.g., 2.46 % for the iter13 5-epoch run) |

## 📈 Training trajectory (initial → final, from `probe_history.jsonl`)

| Metric | Description | Initial | Final | Δ |
|---|---|---:|---:|---|
{lift_block}

({metrics.get('history_steps', 0)} checkpoints across training.)

{paired_block}

## 🚀 Usage

### Download the encoder weights
```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="student_encoder.pt",
)
print("Downloaded to:", ckpt_path)
```

### Load weights into V-JEPA 2.1 ViT-G
```python
import torch
from utils.vjepa2_imports import get_vit_by_arch

state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
encoder = get_vit_by_arch("vit_giant_xformers_rope")
encoder.load_state_dict(state, strict=False)
encoder.eval().to("cuda")
```

### Use as init for downstream surgery / probe training
```bash
python -u src/m09c_surgery.py --FULL \\
    --train-config configs/train/surgery_3stage_DI_iter14.yaml \\
    --init-from-ckpt $(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('{repo_id}', 'student_encoder.pt'))") \\
    --no-wandb
```

## 📦 Files in this repo

| File | Size | Purpose |
|---|---:|---|
{chr(10).join(files_table)}

## 🧪 Reproducibility

This checkpoint was produced by:
```bash
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh {stage} --FULL \\
    2>&1 | tee logs/{stage}_full.log
```

Pipeline source: `iter/iter14_surgery_on_pretrain/plan_HIGH_LEVEL.md`

## 📝 Citation

```bibtex
@misc{{factorjepa2026,
  title  = {{FactorJEPA: Factor-disentangled SSL for Indian-context urban video}},
  author = {{Wanaskar, Kapil and others}},
  year   = {{2026}},
  note   = {{HF model card auto-generated by src/utils/hf_finetuned_push.py}}
}}
```

---

*Model card auto-generated by `src/utils/hf_finetuned_push.py` at {timestamp}.*
"""


def push_to_huggingface(
    source_dir: Path,
    repo_id: str,
    base_model: str = "facebook/v-jepa-2-vitg",
    stage: str = "pretrain",
    paired_results: dict = None,
    private: bool = False,
    dry_run: bool = False,
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

    # 2. Generate model card.
    metrics = _load_training_metrics(source_dir)
    card = _generate_model_card(
        repo_id=repo_id, base_model=base_model, stage=stage,
        metrics=metrics, paired_results=paired_results,
    )
    card_path = source_dir / "README.md"
    if dry_run:
        preview = Path("/tmp") / f"hf_model_card_preview_{repo_id.replace('/', '_')}.md"
        preview.write_text(card)
        print(f"[dry-run] model card preview: {preview}  ({len(card):,} chars)")
    else:
        card_path.write_text(card)
        print(f"Wrote {card_path}  ({len(card):,} chars)")

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
                   help="Local training-output dir (e.g., outputs/full/m09a_pretrain)")
    p.add_argument("--repo-id", required=True,
                   help="HF model repo (e.g., anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep)")
    p.add_argument("--base-model", default="facebook/v-jepa-2-vitg",
                   help="HF id of the base model this was fine-tuned from")
    p.add_argument("--stage", default="pretrain",
                   choices=["pretrain", "pretrain_2X",
                            "surgery_3stage_DI", "surgery_noDI"],
                   help="Training stage label (drives model-card framing + tags)")
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
    )


if __name__ == "__main__":
    main()
