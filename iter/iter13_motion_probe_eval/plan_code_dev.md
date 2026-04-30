# 🛠️ iter13 m06d — Detailed Coding Plan

> ## 🎯 Paper goal — three priorities (durable context for all iter13 work)
>
> 🥇 **P1**: `vjepa_frozen` outperforms `dinov2_frozen` (and DINOv3 if checkpoint available) on Meta's published motion-centric benchmark (SSv2 attentive probe, target **≥ +20 pp**).
>
> 🥈 **P2**: `vjepa_explora` outperforms `vjepa_frozen` on the same benchmark.
>
> 🥉 **P3**: `vjepa_surgery` outperforms `vjepa_explora` on the same benchmark.
>
> ---
>
> ## 📍 This document's scope
>
> Tactical build for **Priority 1 only** — the gate that unblocks P2 / P3. We additionally exercise the **same probe protocol on our own Indian-action 10K pool** (`data/eval_10k_local`) so a passing P1 on Meta-SSv2 is corroborated by V-JEPA also winning on our domain. Three modules below (`m06d_action_probe.py` / `m06d_motion_cos.py` / `m06d_future_mse.py`) implement the V-JEPA-aligned measurements per `analysis.md` Q2. All three are **required**, none optional.

---

## 📋 Status legend

| Emoji | Meaning |
|:-:|:--|
| ⬜ | Pending — not started |
| 🔄 | In progress |
| 🚧 | Partial / blocked |
| ✅ | Completed & verified |
| ❌ | Failed — needs rework |
| 🔥 | Critical / blocking gate |

---

## 🥇 Priority 1 only — `vjepa_frozen` vs `dinov2_frozen`

> Priorities 2 (explora > frozen) and 3 (surgery > explora) are **out of scope for this build** — they wait until Priority 1 PASSes.

### The 3 measurements (per `analysis.md` Q2)

| # | Metric | Module | What it tests |
|:-:|:--|:--|:--|
| 1️⃣ | **Linear / 4-layer attentive probe accuracy** on Indian action classes | 🔥 `src/m06d_action_probe.py` | "The metric V-JEPA was designed to win on" |
| 2️⃣ | **Future-frame latent prediction MSE** | `src/m06d_future_mse.py` | V-JEPA's pretraining objective applied forward-only |
| 3️⃣ | **Per-clip motion-feature cosine similarity** to held-out same-action clips | `src/m06d_motion_cos.py` | Proxy for the SSv2-style motion test on our domain |

---

## 📦 Module roster — 7 files (3 new modules + 2 new utils + 1 util addition + 1 orchestrator)

```text
src/utils/action_labels.py      🆕 NEW shared helper            ~190 LoC  CPU
src/utils/frozen_features.py    🆕 NEW shared loaders+extractor ~280 LoC  GPU
src/utils/vjepa2_imports.py     ✏️ ADD get_attentive_classifier() +22 LoC  CPU
src/m06d_action_probe.py        🔥 PRIMARY GATE                ~500 LoC  GPU
src/m06d_motion_cos.py          (#3 motion proxy)              ~310 LoC  GPU+CPU
src/m06d_future_mse.py          (#2 V-JEPA-only diagnostic)    ~360 LoC  GPU
scripts/run_m06d.sh             🆕 9-stage thin orchestrator   ~200 LoC  shell
```

### 🧱 Architectural summary

```text
        ┌──────────────────────────────────────┐
        │   utils/action_labels.py  (CPU)      │  shared 3/4-class derivation, splits
        │   utils/frozen_features.py (GPU)     │  shared encoder loaders + extractor
        │   utils/vjepa2_imports.py (CPU)      │  + get_attentive_classifier()
        └──────────────────────────────────────┘
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
   m06d_action_probe.py  m06d_motion_cos.py  m06d_future_mse.py
   (4-stage gate)        (3-stage motion proxy) (2-stage V-JEPA-only)
                        │
                        ▼
                scripts/run_m06d.sh
   (9-stage orchestrator: labels → 2× features → 2× train → 🔥P1 GATE
                          → motion features × 2 → cosine × 2 → motion Δ
                          → future_mse forward → future_mse Δ)
```

> 🔧 `frozen_features.py` factored out 2026-04-30 to give `m06d_motion_cos.py` a real fresh-extract path (no `sys.exit` deferral) without violating rule 32 (no cross-imports between m*.py). Single source of truth for `ENCODERS` + V-JEPA / DINOv2 loaders + decode + forward + producer-consumer extraction loop.

---

## 📂 Dataset — `data/eval_10k_local` (10K clips, all 115K-combinations sampled)

| Field | Value |
|:--|:--|
| Source | `data/eval_10k.json` (10K clip_keys, video-level uniform via `m00c_sample_subset.py` over 681 source videos) |
| Local TARs | `data/eval_10k_local/subset-{00000..00009}.tar` |
| VLM tags | `data/eval_10k_local/tags.json` |
| Section schema | `<section>/<video_id>/<file>` — mirrors `m00b_fetch_durations.py:43-77` |

### 📊 Real activity distribution

| Path-derived class | Count | % | ≥ 50 / class? |
|:--|:-:|:-:|:-:|
| 🚶 walking | 5,564 | 55.6 % | ✅ |
| 🚗 driving | 3,053 | 30.5 % | ✅ |
| 🚁 drone | 1,334 | 13.3 % | ✅ |
| 🏛️ monument | 49 | 0.5 % | 🟡 marginal |

**Default = 3-class (Path A)**. `--enable-monument-class` switches to 4-class with VLM `scene_type=heritage_tourist` enrichment.

---

# 🧰 File 1 — `src/utils/action_labels.py` (NEW)

> Status: ✅ shipped 2026-04-30 (~190 LoC) · CPU-only · self-test PASS on `data/eval_10k.json` → 9,951 clips / 3 classes / 70-15-15 stratified split

## Purpose

Single source-of-truth for path-based 3/4-class action label derivation. Consumed by all three `m06d_*.py` modules. Lives in `utils/` to avoid cross-imports between `m*.py` (CLAUDE.md rule 32).

## Module header

```python
"""Path-based action-label derivation for m06d_* modules. CPU-only.
Mirrors m00b_fetch_durations.py extract_all_videos section schema.

USAGE (called by m06d_*.py, never directly via CLI):
    from utils.action_labels import (
        parse_action_from_clip_key, load_subset_with_labels,
        stratified_split, write_action_labels_json, load_action_labels,
    )
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from utils.checkpoint import save_json_checkpoint, load_json_checkpoint
```

## Module constants

```python
# Path activity (from clip_key) → semantic class. Must match m00b extract_all_videos buckets.
PATH_TO_CLASS_3CLASS: dict[str, str] = {
    "walking":  "walking",
    "rain":     "walking",   # tier2 rain bucket = rainy walking tour
    "drive":    "driving",
    "drone":    "drone",
}

# Stable ID order for label encoding (alphabetical → reproducible across runs).
CLASS_NAMES_3CLASS: list[str] = ["driving", "drone", "walking"]                       # n=3
CLASS_NAMES_4CLASS: list[str] = ["driving", "drone", "monument", "walking"]           # n=4

# scene_type override for 4-class mode — VLM tag value that triggers "monument".
HERITAGE_SCENE_TYPE: str = "heritage_tourist"
```

## Public function specs

### `parse_action_from_clip_key`

```python
def parse_action_from_clip_key(
    clip_key: str,
    *,
    enable_monument: bool,
    tags: dict | None = None,
) -> str | None:
    """Derive 3- or 4-class action label from a clip_key (+ optional VLM tags).

    Args:
        clip_key: e.g. "tier1/mumbai/walking/<vid>/<vid>-007.mp4"
        enable_monument: if True, route monuments/* AND tags[clip_key].scene_type=="heritage_tourist" → "monument"
        tags: optional {clip_key: tag_record} lookup; required when enable_monument=True
              and the route is via VLM tag rather than path prefix.

    Returns:
        One of {"driving", "drone", "walking"} (3-class default) or
        adds "monument" (4-class with enable_monument=True), or None for
        monuments/* clips when enable_monument=False (caller filters them out).

    Raises:
        ValueError: clip_key has unrecognized prefix or activity (FAIL-LOUD per CLAUDE.md).

    Examples:
        >>> parse_action_from_clip_key("tier1/mumbai/walking/abc/abc-001.mp4", enable_monument=False)
        'walking'
        >>> parse_action_from_clip_key("tier2/jaipur/rain/abc/abc-002.mp4", enable_monument=False)
        'walking'
        >>> parse_action_from_clip_key("monuments/red_fort_delhi/xyz/xyz-001.mp4", enable_monument=False)
        None
        >>> parse_action_from_clip_key("monuments/red_fort_delhi/xyz/xyz-001.mp4", enable_monument=True)
        'monument'
        >>> parse_action_from_clip_key("tier1/mumbai/walking/abc/abc-001.mp4",
        ...                            enable_monument=True,
        ...                            tags={"tier1/mumbai/walking/abc/abc-001.mp4": {"scene_type": "heritage_tourist"}})
        'monument'
    """
    # 1. Split path
    parts = clip_key.split("/")
    if not parts:
        raise ValueError(f"Empty clip_key: {clip_key!r}")

    # 2. Monument override paths
    if enable_monument and parts[0] == "monuments":
        return "monument"
    if enable_monument and tags is not None:
        rec = tags.get(clip_key)
        if isinstance(rec, dict) and rec.get("scene_type") == HERITAGE_SCENE_TYPE:
            return "monument"

    # 3. monuments/ without enable_monument → return None (caller filters out)
    if parts[0] == "monuments":
        return None

    # 4. Path-based derivation (3-class)
    if parts[0] in ("tier1", "tier2"):
        if len(parts) < 3:
            raise ValueError(f"tier{{1,2}} clip_key missing activity segment: {clip_key!r}")
        activity = parts[2]
    elif parts[0] == "goa":
        if len(parts) < 2:
            raise ValueError(f"goa clip_key missing activity segment: {clip_key!r}")
        activity = parts[1]
    else:
        raise ValueError(f"Unrecognized clip_key prefix '{parts[0]}': {clip_key!r}")

    cls = PATH_TO_CLASS_3CLASS.get(activity)
    if cls is None:
        raise ValueError(f"Unrecognized activity '{activity}' in clip_key: {clip_key!r}")
    return cls
```

### `load_subset_with_labels`

```python
def load_subset_with_labels(
    subset_path: str | Path,
    tags_path: str | Path | None,
    *,
    enable_monument: bool,
) -> list[dict]:
    """Load eval subset JSON + tags, return per-clip records with action labels.

    Args:
        subset_path: data/eval_10k.json
        tags_path:   data/eval_10k_local/tags.json (required if enable_monument=True)
        enable_monument: 4-class mode toggle.

    Returns:
        List[{
            "clip_key": str,
            "class":    str,        # one of CLASS_NAMES_*
            "class_id": int,        # alphabetical index within active class set
        }]
        Clips that derive None (monuments/* with enable_monument=False) are dropped silently.

    Raises:
        FileNotFoundError: subset or tags missing.
        ValueError:        any clip_key fails parse_action_from_clip_key.
    """
    subset_path = Path(subset_path)
    if not subset_path.exists():
        sys.exit(f"FATAL: --eval-subset not found: {subset_path}")
    subset = json.loads(subset_path.read_text())
    clip_keys = subset["clip_keys"]   # fail-loud — no .get(default)

    tags = None
    if enable_monument:
        if tags_path is None:
            sys.exit("FATAL: --tags-json required when --enable-monument-class is set")
        tags_path = Path(tags_path)
        if not tags_path.exists():
            sys.exit(f"FATAL: --tags-json not found: {tags_path}")
        tags_list = json.loads(tags_path.read_text())
        # tags.json schema: list of dicts with 'section'/'video_id'/'source_file' fields
        # build {clip_key: record} lookup matching the format used by m04_vlm_tag.py
        tags = {f"{t['section']}/{t['video_id']}/{t['source_file']}": t for t in tags_list}

    class_names = CLASS_NAMES_4CLASS if enable_monument else CLASS_NAMES_3CLASS
    class_to_id = {c: i for i, c in enumerate(class_names)}

    records: list[dict] = []
    for k in clip_keys:
        cls = parse_action_from_clip_key(k, enable_monument=enable_monument, tags=tags)
        if cls is None:
            continue
        records.append({"clip_key": k, "class": cls, "class_id": class_to_id[cls]})
    return records
```

### `stratified_split`

```python
def stratified_split(
    records: list[dict],
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    seed: int = 99,
) -> dict[str, str]:
    """Stratified-by-class 70/15/15 split. Returns {clip_key: "train"|"val"|"test"}.

    Algorithm:
        1. Group records by class.
        2. Shuffle each class's clip_keys with seed.
        3. Slice train_pct → train, val_pct → val, remainder → test.
        4. Verify ≥ 5 clips/class/split (fail-loud if not).

    Raises:
        ValueError: any class has < 5 clips in any split (sample size too small for BCa).
    """
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_class[r["class"]].append(r["clip_key"])

    splits: dict[str, str] = {}
    for cls, keys in by_class.items():
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)
        n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) < 5:
            raise ValueError(
                f"Class '{cls}' has only n={n} → train={n_train}/val={n_val}/test={n_test}; "
                f"each split must have ≥5 clips for BCa CI to be meaningful."
            )
        for k in keys[:n_train]:
            splits[k] = "train"
        for k in keys[n_train:n_train + n_val]:
            splits[k] = "val"
        for k in keys[n_train + n_val:]:
            splits[k] = "test"
    return splits
```

### `write_action_labels_json`

```python
def write_action_labels_json(
    records: list[dict],
    splits: dict[str, str],
    output_path: str | Path,
) -> dict:
    """Atomically write {clip_key: {class, class_id, split}} to JSON.

    Side-effects:
        - Atomic write via save_json_checkpoint (fsync + os.replace).
        - Writes a sibling class_counts.json with {"class": {"train": n, "val": n, "test": n}}.

    Returns:
        The labels dict that was written (for in-memory chaining).
    """
    out: dict[str, dict] = {}
    for r in records:
        k = r["clip_key"]
        out[k] = {"class": r["class"], "class_id": r["class_id"], "split": splits[k]}
    output_path = Path(output_path)
    save_json_checkpoint(out, output_path)

    # Diagnostic class-count table
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for k, info in out.items():
        counts[info["class"]][info["split"]] += 1
    save_json_checkpoint(dict(counts), output_path.parent / "class_counts.json")
    return out
```

### `load_action_labels`

```python
def load_action_labels(labels_path: str | Path) -> dict[str, dict]:
    """Reverse of write_action_labels_json. Fail-loud if missing."""
    labels_path = Path(labels_path)
    if not labels_path.exists():
        sys.exit(f"FATAL: action_labels.json not found: {labels_path} — "
                 f"run --stage labels first")
    return load_json_checkpoint(labels_path)
```

## 🧪 Self-test (CLI entry point — optional but useful)

```python
if __name__ == "__main__":
    # CPU sanity: derive labels from data/eval_10k.json, print class counts.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval-subset", type=Path, required=True)
    p.add_argument("--tags-json", type=Path, default=None)
    p.add_argument("--enable-monument-class", action="store_true")
    args = p.parse_args()

    records = load_subset_with_labels(args.eval_subset, args.tags_json,
                                      enable_monument=args.enable_monument_class)
    splits = stratified_split(records)
    counts = Counter(r["class"] for r in records)
    print(f"Total: {len(records)} clips ({len(counts)} classes)")
    for cls, n in sorted(counts.items()):
        n_train = sum(1 for r in records if r["class"] == cls and splits[r["clip_key"]] == "train")
        n_val   = sum(1 for r in records if r["class"] == cls and splits[r["clip_key"]] == "val")
        n_test  = sum(1 for r in records if r["class"] == cls and splits[r["clip_key"]] == "test")
        print(f"  {cls:10s}: total={n}  train={n_train}  val={n_val}  test={n_test}")
```

## ⚠️ Edge cases

| Case | Behavior |
|:--|:--|
| `clip_key=""` (empty) | `ValueError` with the empty string echoed |
| `clip_key="randomstring"` (no `/`) | `ValueError("Unrecognized clip_key prefix")` |
| `clip_key="monuments/X/Y/Z.mp4"` + `enable_monument=False` | returns `None` → silently dropped by `load_subset_with_labels` |
| `tags=None` + `enable_monument=True` + path is monuments/ | OK — path-based override fires first |
| `tags=None` + `enable_monument=True` + path is tier1/walking | OK — falls through to path-based 3-class |
| Class with < 5 clips in any split | `ValueError` from `stratified_split` |

---

# 🔌 File 2 — `src/utils/vjepa2_imports.py` (ADD `get_attentive_classifier`)

> Status: ✅ shipped 2026-04-30 (+22 LoC patch — `get_attentive_classifier` + `get_attentive_pooler` + 1-line `importlib.import_module("src.models.attentive_pooler")` in `_ensure_loaded_base`). REPL-verified on `venv_walkindia`: 130.23 M params, forward `(B=2, T=4608, D=1664) → (2, 3)`.

## Purpose

Expose Meta's `AttentiveClassifier` (defined in `deps/vjepa2/src/models/attentive_pooler.py`) through the existing namespace shim so `m06d_action_probe.py` can import it without polluting `sys.modules` with `src.*` collisions.

## Diff to apply

**Insertion site**: end of "Shared (works with both 2.0 and 2.1)" section (after `get_apply_masks`).

```python
# ── Attentive probe head (used by evals + m06d_action_probe) ─────────

def get_attentive_classifier():
    """Meta's AttentiveClassifier from deps/vjepa2/src/models/attentive_pooler.py.

    Wraps an N-layer AttentivePooler + Linear head for classification.
    Kwargs: embed_dim, num_classes, depth (=N pool layers), num_heads,
            mlp_ratio, complete_block, use_activation_checkpointing.

    This is the EXACT module Meta uses for SSv2 / K400 / Diving-48 / EK100
    attentive probes (see deps/vjepa2/evals/video_classification_frozen/).
    Bit-identical to V-JEPA 2.1 published numbers when train recipe matches.
    """
    _ensure_loaded_base()
    return sys.modules["src.models.attentive_pooler"].AttentiveClassifier


def get_attentive_pooler():
    """Same family — returns the bare AttentivePooler (no Linear head).

    Use when you want pooled features only (e.g., motion_cos doesn't need a head).
    """
    _ensure_loaded_base()
    return sys.modules["src.models.attentive_pooler"].AttentivePooler
```

## Required upstream change

`_ensure_loaded_base()` already loads the modules `AttentiveClassifier` depends on (`src.models.utils.modules`, `src.utils.tensors`). Add **one** line to the `try:` block:

```python
# In _ensure_loaded_base(), inside the try: block (line ~52)
importlib.import_module("src.models.attentive_pooler")    # NEW — adds AttentivePooler + AttentiveClassifier
```

## 🧪 Verification

```python
# REPL test (run after edit):
from utils.vjepa2_imports import get_attentive_classifier, get_attentive_pooler
cls = get_attentive_classifier()
clf = cls(embed_dim=1664, num_classes=3, depth=4, num_heads=16, mlp_ratio=4.0)
print(sum(p.numel() for p in clf.parameters()))   # expect ~50–80M params for ViT-G dim
```

## ⚠️ Failure mode

If Meta renames `attentive_pooler.py` or moves `AttentiveClassifier` out of `src.models`, the import inside `_ensure_loaded_base()` raises `ImportError`. This is **caught at preflight B71** (see Module 1) — m06d exits with a clear "Meta repo schema drift, pin a different commit" message rather than silently broken behavior.

---

# 🔥 File 3 — `src/m06d_action_probe.py` (PRIMARY GATE) — DETAILED CODING PLAN

> Status: 🚧 partially shipped 2026-04-30 (~590 LoC) · GPU · 4-stage pipeline (`labels` / `features` / `train` / `paired_delta`). Stage 1 (`labels`) verified end-to-end on `data/eval_10k.json`. Stages 2–4 coded but unexercised (need GPU run).

## Module header

```python
"""Indian action attentive probe (Priority 1 gate). GPU-only.

Stages:
  labels        — derive 3/4-class action labels + 70/15/15 split (CPU)
  features      — extract frozen spatiotemporal token features per encoder (GPU)
  train         — train AttentiveClassifier head on cached features (GPU, small)
  paired_delta  — paired BCa Δ between V-JEPA and DINOv2 (CPU)

USAGE (full sequence — every path arg required, no defaults):
    # Stage 1: labels (CPU, ~1 min)
    python -u src/m06d_action_probe.py --SANITY \\
        --stage labels \\
        --eval-subset data/eval_10k.json \\
        --tags-json data/eval_10k_local/tags.json \\
        --output-root outputs/sanity/m06d_action_probe \\
        --cache-policy 1 \\
        2>&1 | tee logs/m06d_action_probe_labels_sanity.log

    # Stage 2: features (GPU, ~30 min × 2 encoders)
    python -u src/m06d_action_probe.py --FULL \\
        --stage features \\
        --encoder vjepa_2_1_frozen \\
        --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \\
        --eval-subset data/eval_10k.json \\
        --local-data data/eval_10k_local \\
        --output-root outputs/full/m06d_action_probe \\
        --cache-policy 1 \\
        2>&1 | tee logs/m06d_action_probe_features_vjepa.log

    # Stage 3: train probe (GPU, ~15 min × 2 encoders)
    python -u src/m06d_action_probe.py --FULL \\
        --stage train \\
        --encoder vjepa_2_1_frozen \\
        --output-root outputs/full/m06d_action_probe \\
        --cache-policy 1

    # Stage 4: paired Δ (CPU, ~5 min, BCa 10K bootstrap)
    python -u src/m06d_action_probe.py --FULL \\
        --stage paired_delta \\
        --output-root outputs/full/m06d_action_probe \\
        --cache-policy 1
"""
```

## Imports

```python
import argparse, json, os, queue, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.action_labels import (
    load_subset_with_labels, stratified_split, write_action_labels_json,
    load_action_labels, CLASS_NAMES_3CLASS, CLASS_NAMES_4CLASS,
)
from utils.bootstrap import paired_bca, bootstrap_ci
from utils.cache_policy import (
    add_cache_policy_arg, resolve_cache_policy_interactive,
    guarded_delete, wipe_output_dir,
)
from utils.checkpoint import (
    save_array_checkpoint, save_json_checkpoint, load_json_checkpoint,
)
from utils.config import (
    add_subset_arg, add_local_data_arg, get_pipeline_config,
    check_gpu, get_total_clips, get_sanity_clip_limit,
)
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.gpu_batch import (
    AdaptiveBatchSizer, cleanup_temp, cuda_cleanup, compute_batch_sizes,
)
from utils.progress import make_pbar
from utils.video_io import decode_video_bytes
from utils.vjepa2_imports import get_attentive_classifier, get_vit_by_arch
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb
```

## Module constants

```python
_PCFG = get_pipeline_config()
NUM_FRAMES        = 16                                          # OQ2 default
PATCH_SIZE        = 16
TUBELET_SIZE      = 2
DECODE_WORKERS    = _PCFG["streaming"]["decode_workers_embed"]
PREFETCH_QUEUE    = _PCFG["streaming"]["prefetch_queue_embed"]
CHECKPOINT_EVERY  = _PCFG["streaming"]["checkpoint_every"]

# Encoder catalog (priority 1 only; explora/surgery added in priorities 2/3 turns)
ENCODERS = {
    "vjepa_2_1_frozen": {
        "kind": "vjepa",
        "arch": "vit_gigantic_xformers",
        "crop": 384,
        "embed_dim": 1664,
    },
    "dinov2": {
        "kind": "dinov2",
        "model_id": "facebook/dinov2-with-registers-giant",
        "embed_dim": 1536,
    },
}
```

## CLI builder

```python
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Indian action attentive probe (m06d action_probe — priority 1 gate)")
    # mode flags
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    # stage dispatch
    p.add_argument("--stage", required=True,
                   choices=["labels", "features", "train", "paired_delta"])
    # encoder (required for features + train; ignored for labels + paired_delta)
    p.add_argument("--encoder", choices=list(ENCODERS), default=None)
    p.add_argument("--encoder-ckpt", type=Path, default=None,
                   help="Required for --stage features when encoder is V-JEPA")
    # paths (no defaults — CLAUDE.md no-default rule)
    p.add_argument("--eval-subset", type=Path, default=None,
                   help="Required for --stage labels and features")
    p.add_argument("--tags-json", type=Path, default=None,
                   help="Required when --enable-monument-class is set")
    add_local_data_arg(p)        # registers --local-data
    p.add_argument("--output-root", type=Path, required=True,
                   help="e.g. outputs/full/m06d_action_probe")
    # task knobs
    p.add_argument("--enable-monument-class", action="store_true",
                   help="Path B (4-class) — enrich monument via scene_type=heritage_tourist")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--probe-lr", type=float, default=5e-4)
    p.add_argument("--probe-wd", type=float, default=0.05)
    p.add_argument("--probe-depth", type=int, default=4,
                   help="N attentive-pool layers (V-JEPA 2.1 published: 4)")
    p.add_argument("--seed", type=int, default=99)
    # shared infra
    add_cache_policy_arg(p)
    add_wandb_args(p)
    return p
```

## Stage dispatch

```python
def main() -> None:
    args = build_parser().parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        sys.exit("ERROR: specify --SANITY, --POC, or --FULL")
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    args.output_root.mkdir(parents=True, exist_ok=True)
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb = init_wandb(f"m06d_action_probe_{args.stage}", mode,
                    config=vars(args), enabled=not args.no_wandb)

    if args.stage == "labels":
        run_labels_stage(args, wb)
    elif args.stage == "features":
        run_features_stage(args, wb)
    elif args.stage == "train":
        run_train_stage(args, wb)
    elif args.stage == "paired_delta":
        run_paired_delta_stage(args, wb)
    finish_wandb(wb)
```

## Stage 1 — `run_labels_stage(args, wb)` (CPU, ~1 min)

```python
def run_labels_stage(args, wb) -> None:
    """Derive action_labels.json + class_counts.json. Fails loud on label issues.

    Pipeline:
        1. Validate paths.
        2. Load subset + tags via utils.action_labels.
        3. Stratified 70/15/15 split (seed=99 by default).
        4. Atomic write of action_labels.json + class_counts.json.
        5. Pre-flight B74: ≥30 train + ≥5 val + ≥5 test per class.
    """
    if args.eval_subset is None:
        sys.exit("FATAL: --stage labels requires --eval-subset")

    labels_path = args.output_root / "action_labels.json"
    if labels_path.exists() and args.cache_policy == "1":
        print(f"  [keep] {labels_path} present — skipping (use --cache-policy 2 to redo)")
        return
    guarded_delete(labels_path, args.cache_policy, "action_labels.json")
    guarded_delete(args.output_root / "class_counts.json", args.cache_policy, "class_counts.json")

    records = load_subset_with_labels(args.eval_subset, args.tags_json,
                                      enable_monument=args.enable_monument_class)
    print(f"Loaded {len(records)} labeled clips from {args.eval_subset}")
    splits = stratified_split(records, seed=args.seed)
    write_action_labels_json(records, splits, labels_path)

    # Preflight B74: per-class minimums
    counts = load_json_checkpoint(args.output_root / "class_counts.json")
    for cls, c in counts.items():
        if c["test"] < 5 or c["val"] < 5:
            sys.exit(f"FATAL: class '{cls}' has val={c['val']}/test={c['test']} (need ≥5 each)")
        if c["train"] < 30:
            print(f"  WARN: class '{cls}' has train={c['train']} (recommended ≥30)")
    log_metrics(wb, {"n_clips_labeled": len(records),
                     "n_classes": len(counts)})
    print(f"Wrote: {labels_path}  +  class_counts.json")
```

## Stage 2 — `run_features_stage(args, wb)` (GPU, ~30 min/encoder)

```python
def run_features_stage(args, wb) -> None:
    """Extract frozen spatiotemporal token features for one encoder, write per-split .npy.

    Outputs (per encoder):
        <output_root>/<encoder>/features_train.npy   shape: (N_train, n_tokens, D)
        <output_root>/<encoder>/features_val.npy
        <output_root>/<encoder>/features_test.npy
        <output_root>/<encoder>/clip_keys_<split>.npy   (str object array, aligned)

    Notes:
        - V-JEPA: model.forward(pixels) returns (B, n_spatiotemporal_tokens, D)
          where n_tokens = (crop/patch)^2 × (T/tubelet) — for 384/16/16/2 = 24*24*8 = 4608.
        - DINOv2 video recipe: process each of T frames separately → (B*T, n_spatial, D),
          reshape to (B, T*n_spatial, D) — matches V-JEPA 2 paper §4.1 'tile + temporal pool'.
        - Sub-batched via AdaptiveBatchSizer with per-clip OOM retry.
        - Resume via per-split checkpoint files (`.m06d_features_<split>_<enc>_ckpt.npz`).
    """
    if args.encoder is None or args.eval_subset is None or args.local_data is None:
        sys.exit("FATAL: --stage features requires --encoder, --eval-subset, --local-data")
    if ENCODERS[args.encoder]["kind"] == "vjepa" and args.encoder_ckpt is None:
        sys.exit("FATAL: V-JEPA encoder requires --encoder-ckpt")

    check_gpu()
    cleanup_temp()
    ensure_local_data(args)

    labels = load_action_labels(args.output_root / "action_labels.json")
    enc_dir = args.output_root / args.encoder
    enc_dir.mkdir(parents=True, exist_ok=True)

    # Load model + processor (delegates to encoder-specific helper)
    if ENCODERS[args.encoder]["kind"] == "vjepa":
        model, preprocess, embed_dim = _load_vjepa_2_1_frozen(args.encoder_ckpt, args.num_frames)
    elif ENCODERS[args.encoder]["kind"] == "dinov2":
        model, preprocess, embed_dim = _load_dinov2_frozen()
    else:
        sys.exit(f"FATAL: unknown encoder kind: {ENCODERS[args.encoder]['kind']}")

    # Bucket clip_keys by split
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for k, info in labels.items():
        by_split[info["split"]].append(k)

    # Per-split extraction loop
    for split, keys in by_split.items():
        out_features = enc_dir / f"features_{split}.npy"
        out_keys     = enc_dir / f"clip_keys_{split}.npy"
        if out_features.exists() and out_keys.exists() and args.cache_policy == "1":
            print(f"  [keep] {split}: features cached → skipping")
            continue
        guarded_delete(out_features, args.cache_policy, f"features_{split}")
        guarded_delete(out_keys, args.cache_policy, f"clip_keys_{split}")

        feats, ordered_keys = _extract_features_for_split(
            model, preprocess, args, split, keys, enc_dir,
        )                                                            # → (N, n_tokens, D), [keys]
        save_array_checkpoint(feats, out_features)
        np.save(out_keys, np.array(ordered_keys, dtype=object))
        print(f"  {split}: wrote {out_features} {feats.shape}")
        log_metrics(wb, {f"features_{split}_n": feats.shape[0],
                         f"features_{split}_dim": feats.shape[-1]})
```

### Key sub-helpers (signatures only — bodies follow `m05_vjepa_embed.py:580-700` patterns)

```python
def _load_vjepa_2_1_frozen(ckpt_path: Path, num_frames: int) -> tuple:
    """Load V-JEPA 2.1 ViT-G frozen encoder (target_encoder weights, bf16).
       Returns (model, preprocess_fn, embed_dim=1664).
       Mirrors m05_vjepa_embed.py:629-670 — same dtype, same `use_rope=True`.
    """

def _load_dinov2_frozen() -> tuple:
    """Load HF facebook/dinov2-with-registers-giant in fp16+FA2.
       Returns (model, processor, embed_dim=1536).
       Mirrors m05b_baselines.py:368-388.
    """

def _extract_features_for_split(
    model, preprocess, args, split: str, keys: list[str], enc_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Producer-consumer pattern with iter_clips_parallel + AdaptiveBatchSizer.
       Saves intermediate .npz checkpoint every CHECKPOINT_EVERY clips.

       Returns:
           features: (N, n_tokens, D) float32
           ordered_keys: [clip_key, ...] aligned with features rows
       Throws torch.cuda.OutOfMemoryError if min sub-batch still OOMs (no fallback).
    """
```

## Stage 3 — `run_train_stage(args, wb)` (GPU, ~15 min/encoder, small head)

```python
def run_train_stage(args, wb) -> None:
    """Train AttentiveClassifier on cached features. Outputs:
        <output_root>/<encoder>/probe.pt          (best-val checkpoint)
        <output_root>/<encoder>/train_log.jsonl   (per-epoch loss + val_acc)
        <output_root>/<encoder>/test_predictions.npy (N_test,) ∈ {0,1} per-clip top-1
        <output_root>/<encoder>/test_clip_keys.npy   (paired with test_predictions)
        <output_root>/<encoder>/test_metrics.json  (acc + 95 % BCa CI)

    Recipe (V-JEPA 2.1 paper §4.2 conventions):
        AdamW(lr=5e-4, wd=0.05) · cosine schedule · 10 % warmup · 50 epochs ·
        cross-entropy · seed=99 · best-val-acc model selection.
    """
    if args.encoder is None:
        sys.exit("FATAL: --stage train requires --encoder")
    check_gpu()
    cleanup_temp()

    enc_dir = args.output_root / args.encoder
    labels = load_action_labels(args.output_root / "action_labels.json")
    class_names = (CLASS_NAMES_4CLASS if args.enable_monument_class
                   else CLASS_NAMES_3CLASS)

    # Load cached features
    feats_train = np.load(enc_dir / "features_train.npy")
    feats_val   = np.load(enc_dir / "features_val.npy")
    feats_test  = np.load(enc_dir / "features_test.npy")
    keys_train  = np.load(enc_dir / "clip_keys_train.npy", allow_pickle=True)
    keys_val    = np.load(enc_dir / "clip_keys_val.npy",   allow_pickle=True)
    keys_test   = np.load(enc_dir / "clip_keys_test.npy",  allow_pickle=True)
    y_train = np.array([labels[str(k)]["class_id"] for k in keys_train])
    y_val   = np.array([labels[str(k)]["class_id"] for k in keys_val])
    y_test  = np.array([labels[str(k)]["class_id"] for k in keys_test])

    d_in = feats_train.shape[-1]
    n_classes = len(class_names)

    # Preflight B71
    AttentiveClassifier = get_attentive_classifier()
    probe = AttentiveClassifier(embed_dim=d_in, num_classes=n_classes,
                                depth=args.probe_depth, num_heads=16,
                                mlp_ratio=4.0, complete_block=True).cuda()

    # Train loop (signature only — body straightforward DataLoader + AdamW + cosine)
    best_val_acc, best_state = _train_attentive_classifier(
        probe, feats_train, y_train, feats_val, y_val,
        args, jsonl_path=enc_dir / "train_log.jsonl", wb=wb)
    torch.save(best_state, enc_dir / "probe.pt")

    # Eval on test
    probe.load_state_dict(best_state)
    probe.eval()
    test_correct, test_acc, test_ci = _eval_attentive_classifier(
        probe, feats_test, y_test)                          # → (N,)∈{0,1}, scalar acc, ci dict
    np.save(enc_dir / "test_predictions.npy", test_correct.astype(np.int8))
    np.save(enc_dir / "test_clip_keys.npy", keys_test)
    save_json_checkpoint({
        "encoder": args.encoder, "n_classes": n_classes, "class_names": class_names,
        "n_test": int(len(test_correct)),
        "top1_acc": float(test_acc), "top1_ci": test_ci,
        "best_val_acc": float(best_val_acc),
    }, enc_dir / "test_metrics.json")
    log_metrics(wb, {"test_top1_acc": test_acc,
                     "test_top1_ci_half": test_ci["ci_half"]})
```

### Sub-helpers

```python
def _train_attentive_classifier(probe, X_tr, y_tr, X_val, y_val, args, *, jsonl_path, wb):
    """Standard AdamW + cosine schedule + 10 % warmup + cross-entropy.
       Best-by-val-acc selection. JSONL log per epoch (loss, train_acc, val_acc, lr).
       Sub-batches via AdaptiveBatchSizer if VRAM tight (~10GB max for 4-layer head).
       Returns (best_val_acc, best_state_dict).
    """

def _eval_attentive_classifier(probe, X, y) -> tuple[np.ndarray, float, dict]:
    """Returns (per_clip_correct ∈ {0,1}, mean_acc, BCa_CI) using utils.bootstrap.bootstrap_ci."""
```

## Stage 4 — `run_paired_delta_stage(args, wb)` (CPU, ~5 min)

```python
def run_paired_delta_stage(args, wb) -> None:
    """Paired BCa Δ between V-JEPA and DINOv2 on test split.

    Output: <output_root>/m06d_paired_delta.json

    Preflight B72: test_clip_keys.npy must be byte-identical between the two encoder dirs.
    """
    vj_dir = args.output_root / "vjepa_2_1_frozen"
    dn_dir = args.output_root / "dinov2"
    for d in (vj_dir, dn_dir):
        if not (d / "test_predictions.npy").exists():
            sys.exit(f"FATAL: {d}/test_predictions.npy missing — run --stage train first")

    vj = np.load(vj_dir / "test_predictions.npy").astype(np.float32)
    dn = np.load(dn_dir / "test_predictions.npy").astype(np.float32)
    vj_keys = np.load(vj_dir / "test_clip_keys.npy", allow_pickle=True)
    dn_keys = np.load(dn_dir / "test_clip_keys.npy", allow_pickle=True)
    if not np.array_equal(vj_keys, dn_keys):
        sys.exit("FATAL: test split clip_keys disagree between encoders — re-run features stage")

    delta = vj - dn                                              # (N,) ∈ {-1, 0, +1}
    bca = paired_bca(delta)                                      # mean, ci_lo, ci_hi, p
    out = {
        "metric": "top1_accuracy",
        "n_clips_test": int(len(delta)),
        "vjepa_acc_pct":   float(vj.mean() * 100),
        "dinov2_acc_pct":  float(dn.mean() * 100),
        "delta_pp":        float(delta.mean() * 100),
        "ci_lo_pp":        float(bca["ci_lo"] * 100),
        "ci_hi_pp":        float(bca["ci_hi"] * 100),
        "ci_half_pp":      float(bca["ci_half"] * 100),
        "p_value":         bca["p_value_vs_zero"],
        "gate_pass":       bool(bca["ci_lo"] > 0),
    }
    save_json_checkpoint(out, args.output_root / "m06d_paired_delta.json")
    log_metrics(wb, out)
    print(json.dumps(out, indent=2))
```

## 📥 Pre-flight checks

| # | Check | Stage | Action |
|:-:|:--|:-:|:--|
| ⬜ B70 | `--eval-subset` JSON exists, has `clip_keys` ≥ 100 | labels | `sys.exit(3)` |
| ⬜ B71 | `get_attentive_classifier()` returns class | features+train | `sys.exit(3)` |
| ⬜ B72 | Test split clip_keys identical between encoders | paired_delta | `sys.exit(6)` |
| ⬜ B73 | `--encoder-ckpt` exists, > 1 GB, loadable | features | `sys.exit(3)` |
| ⬜ B74 | After labels: ≥30 train + ≥5 val + ≥5 test per class | labels | warn / `sys.exit(4)` |

---

# 📊 File 4 — `src/m06d_motion_cos.py` — DETAILED CODING PLAN

> Status: ✅ shipped 2026-04-30 (~310 LoC) · GPU+CPU · 3-stage (`features` / `cosine` / `paired_delta`). **Both share-features AND fresh-extract paths implemented** — fresh path delegates to new `utils/frozen_features.py` (~280 LoC factored out of action_probe so both modules use the same loaders/extractor — CLAUDE.md rule 32 compliant). `--share-features` falls back automatically to fresh extract if action_probe cache is absent (no surprise `sys.exit`). Algorithmic unit-test PASS (in-class cos ≈ 0.997, between-class ≈ 0.198, motion_score ∈ [0.75, 0.85]); edge cases PASS (shape mismatch raises, single-class neg=0). Single source-of-truth for `ENCODERS` verified across `frozen_features` / `action_probe` / `motion_cos`.

## Module header

```python
"""Per-clip motion-feature cosine similarity to held-out same-action neighbors.
Proxy for SSv2-style motion test on Indian clips. GPU (features) + CPU (cosine math).

For each test clip q with class c(q):
    pos_sim(q) = mean cos(emb_q, emb_n) for n ∈ test ∧ c(n) = c(q) ∧ n ≠ q
    neg_sim(q) = mean cos(emb_q, emb_n) for n ∈ test ∧ c(n) ≠ c(q)
    motion_score(q) = pos_sim(q) − neg_sim(q)        ∈ [-2, 2]   (higher = better)

Stages:
  features      — extract MEAN-POOLED frozen features (1 vec per clip) [GPU, ~15 min/encoder]
                  (or symlink reuse of m06d_action_probe features via --share-features)
  cosine        — vectorised intra/inter cosine on test split (CPU, ~1 min)
  paired_delta  — paired BCa Δ_motion = motion_score_vjepa − motion_score_dinov2 (CPU)

USAGE: see action_probe sibling — same flag conventions.
"""
```

## Public stages

```python
def run_features_stage(args, wb) -> None:
    """Extract pooled (N, D) features per encoder.

    If --share-features is set AND m06d_action_probe/<encoder>/features_test.npy exists,
    mean-pool over the n_tokens axis to derive (N, D) and skip a re-extraction.
    Otherwise extract fresh from local TARs (saves to <output_root>/<encoder>/pooled_features_test.npy).
    """

def run_cosine_stage(args, wb) -> None:
    """Vectorised intra/inter cosine.

    Inputs:  pooled_features_test.npy (N_test, D) + action_labels.json (test split)
    Outputs: per_clip_motion_cos.npy (N_test,) per-clip motion_score
             intra_inter_ratio.json   {"pos_mean": float, "neg_mean": float,
                                       "score_mean": float, "score_ci": {...}}
    """

def run_paired_delta_stage(args, wb) -> None:
    """Paired BCa Δ_motion. Output: m06d_motion_cos_paired.json."""
```

## Vectorised cosine math (the heart of Stage 2)

```python
def _per_clip_motion_score(emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Vectorised pos/neg cosine. emb shape (N, D), labels shape (N,).
    Returns motion_score per clip, shape (N,).

    Algorithm (no Python for-loop):
        emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)        # L2-normalise
        S = emb_n @ emb_n.T                                              # (N, N) cos sim
        same = labels[:, None] == labels[None, :]                        # bool (N, N)
        np.fill_diagonal(same, False)                                    # exclude self
        diff = ~same
        np.fill_diagonal(diff, False)
        # Mean over True elements (broadcast-safe with float division):
        pos_mean = (S * same).sum(1) / np.clip(same.sum(1), 1, None)
        neg_mean = (S * diff).sum(1) / np.clip(diff.sum(1), 1, None)
        return pos_mean - neg_mean                                        # (N,)
    """
```

> **Memory note**: at N=1500 test clips × 4 bytes = ~9 MB for `S` — well within RAM. At N=10K (if we ever scale): ~400 MB still fine on CPU. For N > 50K we'd switch to chunked computation.

## CLI mirrors action_probe

Same `--SANITY/--POC/--FULL`, same `--stage`, same `--encoder`, same `--cache-policy`. Adds:

```python
p.add_argument("--share-features", action="store_true",
               help="Reuse pooled-features from m06d_action_probe/<encoder>/ if present")
```

## Output

```json
// m06d_motion_cos_paired.json
{
  "metric": "intra_minus_inter_cosine",
  "n_test": 1500,
  "vjepa_score_mean": 0.3142, "vjepa_score_ci": {"ci_lo": 0.31, "ci_hi": 0.32, "ci_half": 0.005},
  "dinov2_score_mean": 0.2284, "dinov2_score_ci": {...},
  "delta_mean": 0.0858, "delta_ci_half": 0.0094, "p_value": 0.0,
  "gate_pass": true
}
```

## Pre-flight

| # | Check | On failure |
|:-:|:--|:--|
| ⬜ B75 | `pooled_features_*.npy` shape matches labels JSON N | `sys.exit(4)` |
| ⬜ B76 | When `--share-features`: action_probe features file exists + same N | warn → fall back to fresh extract |

---

# 🔮 File 5 — `src/m06d_future_mse.py` — DETAILED CODING PLAN

> Status: ✅ shipped 2026-04-30 (~440 LoC) · GPU · **V-JEPA-only**. 2-stage (`forward` / `paired_per_variant`). Loads encoder + predictor + MaskGenerator from V-JEPA 2.1 release `.pt` (predictor params hardcoded from `configs/model/vjepa2_1.yaml`: depth=24, pred_embed_dim=384, num_heads=12, num_mask_tokens=2). Per-clip L1 forward mirrors `utils/training.py:528-534` jepa-loss step (encoder(pixel, masks=[m_enc]) → context, encoder(pixel) → target, apply_masks(h, [m_pred]) → target tokens, predictor(z, [m_enc], [m_pred], mask_index=0) → predicted target tokens, |Δ| mean over (n_pred_tokens, D) per clip). CPU-side smoke-test PASS: imports OK, `_build_mask_gen` returns `_MaskGenerator` instance, `mg(B=2)` returns tensors of shape `(2, 2048)`. GPU-forward path unexercised (deferred to deliberate kickoff alongside Stage 2 of action_probe).

## Module header

```python
"""Future-frame latent prediction MSE. V-JEPA-only diagnostic. GPU.

Splits each 16-frame clip into context (first 8) + target (last 8). Encodes
both with the V-JEPA 2.1 frozen encoder, then asks the V-JEPA predictor to
hallucinate target latents from context latents. Compute L1 (matches V-JEPA's
training loss).

Why V-JEPA only:
  DINOv2 is image-only — no future-frame predictor head. Reporting DINOv2 here
  would require a fresh predictor, which is (a) re-training under the wrong
  objective, (b) not part of DINOv2's published recipe. So this module is a
  HEALTH CHECK for V-JEPA on Indian video, not a paired V-JEPA-vs-DINOv2 test.

Stages:
  forward                — encode context+target, run predictor, dump per-clip MSE [GPU]
  paired_per_variant     — when priorities 2/3 land, compares {frozen, explora,
                           surgery_*} per-clip MSE distributions (CPU, BCa)
"""
```

## Public stages

```python
def run_forward_stage(args, wb) -> None:
    """V-JEPA encoder + predictor on test split. Outputs:
        <output_root>/<vjepa_variant>/per_clip_mse.npy   (N_test,) float32
        <output_root>/<vjepa_variant>/aggregate_mse.json {mse_mean, mse_std, mse_ci, n}

    Pipeline per clip (16 frames):
        1. Decode → (T=16, H, W, 3) uint8.
        2. Split: context = frames[:8], target = frames[8:].
        3. Encode both with target_encoder (V-JEPA frozen), bf16 forward.
        4. Build ATTENTIVE 'predict-target-from-context' mask using V-JEPA's
           native MaskGenerator (deps/vjepa2/src/masks/multiseq_multiblock3d.py).
        5. predictor(z_context, z_target_masked, masks) → predicted target tokens.
        6. L1 distance between predicted and true target tokens, mean over tokens.
        7. per_clip_mse[i] = float(L1.cpu().item())
    """

def run_paired_per_variant_stage(args, wb) -> None:
    """When 2 or more variants exist, compute pairwise BCa Δ on per-clip MSE.

    For priority 1 (V-JEPA frozen only) this stage is informational — it
    reports the single variant + 'dinov2': 'n/a — no future-frame predictor'.
    """
```

## Sub-helpers (signatures)

```python
def _load_vjepa_2_1_for_future(ckpt_path: Path, num_frames: int = 16) -> tuple:
    """Loads:
        target_encoder  — frozen V-JEPA 2.1 ViT-G (target_encoder weights)
        predictor       — V-JEPA 2.1 predictor (predictor weights from same ckpt)
        mask_generator  — multiseq_multiblock3d._MaskGenerator with V-JEPA 2.1 defaults
    Returns (encoder, predictor, mask_gen, embed_dim).
    """

def _compute_one_clip_mse(encoder, predictor, mask_gen, frames, device) -> float:
    """Forward pass for one clip. Returns scalar L1 loss between predicted and true target tokens."""

def _producer_consumer_loop(args, encoder, predictor, mask_gen, keys, output_dir):
    """Mirrors m05_vjepa_embed.py producer-consumer pattern with iter_clips_parallel.
    Saves per_clip_mse.npy incrementally via save_array_checkpoint every 500 clips.
    """
```

## Output (priority 1 only)

```json
// m06d_future_mse_per_variant.json
{
  "metric": "future_frame_l1_loss",
  "by_variant": {
    "vjepa_2_1_frozen":  {"mse_mean": 0.482, "mse_std": 0.041,
                          "mse_ci": {"ci_lo": 0.480, "ci_hi": 0.484, "ci_half": 0.002},
                          "n": 1500},
    "vjepa_2_1_explora":  null,
    "vjepa_2_1_surgical": null,
    "dinov2":            "n/a — no future-frame predictor"
  }
}
```

## Pre-flight

| # | Check | On failure |
|:-:|:--|:--|
| ⬜ B76 | `--encoder-ckpt` contains both `target_encoder` AND `predictor` keys | `sys.exit(3)` |
| ⬜ B77 | mask_generator load returns non-None (Meta repo schema drift detector) | `sys.exit(3)` |

## Failure mode

If V-JEPA 2.1 release ships predictor weights in a SEPARATE file rather than the same `.pt`, expose `--predictor-ckpt <path>` flag (handled in CLI but defaults to "expect inside encoder_ckpt"). FAIL-LOUD if neither path yields predictor weights.

---

## 📤 Output schemas (reproduced for quick reference)

### `m06d_action_probe/m06d_paired_delta.json`

```json
{
  "metric": "top1_accuracy", "n_clips_test": 1500,
  "vjepa_acc_pct": 78.42, "dinov2_acc_pct": 65.18,
  "delta_pp": 13.24, "ci_lo_pp": 11.31, "ci_hi_pp": 15.19, "ci_half_pp": 1.94,
  "p_value": 0.0, "gate_pass": true
}
```

### `m06d_motion_cos/m06d_motion_cos_paired.json`

```json
{ "metric": "intra_minus_inter_cosine", "vjepa_score_mean": 0.3142,
  "dinov2_score_mean": 0.2284, "delta_mean": 0.0858, "delta_ci_half": 0.0094,
  "p_value": 0.0, "gate_pass": true, "n_test": 1500 }
```

### `m06d_future_mse/m06d_future_mse_per_variant.json`

```json
{ "metric": "future_frame_l1_loss",
  "by_variant": { "vjepa_2_1_frozen": {"mse_mean": 0.482, "mse_std": 0.041,
                                       "mse_ci": {...}, "n": 1500},
                  "dinov2": "n/a — no future-frame predictor"} }
```

---

## 🗄️ Cache policy (mirrors `m05` / `m06b`)

| Artifact | `--cache-policy 1` (keep) | `--cache-policy 2` (recompute) |
|:--|:--|:--|
| `action_labels.json` | ♻️ reuse | 🗑️ `guarded_delete()` → re-derive |
| `<encoder>/features_*.npy` | ♻️ reuse if shape matches | 🗑️ re-extract (~30 min/encoder) |
| `<encoder>/probe.pt` + `train_log.jsonl` | ♻️ skip train stage | 🗑️ re-train (~15 min/encoder) |
| `<encoder>/test_predictions.npy` | ♻️ reuse for paired-Δ | 🗑️ regenerate via train stage |
| `m06d_paired_delta.json` | ✏️ overwrite always (cheap) | ✏️ overwrite always |

> 💬 Interactive prompt via `resolve_cache_policy_interactive()` if `--cache-policy` not on CLI. Shells stay thin — pass `--cache-policy 1` explicitly in tmux runs.

---

## 💰 Cost matrix (priority 1, on `data/eval_10k_local`)

| Stage | Module | GPU-h | $ @ Blackwell |
|:--|:--|:-:|:-:|
| labels | action_probe | 0 | 0 |
| features × 2 enc | action_probe | 1.0 | $0.80 |
| train × 2 enc | action_probe | 0.5 | $0.40 |
| paired_delta | action_probe | 0 | 0 |
| features × 2 enc (or reuse) | motion_cos | 0.5 (0 if shared) | $0.40 |
| cosine + paired_delta | motion_cos | 0 | 0 |
| forward × 1 (V-JEPA frozen only) | future_mse | 0.5 | $0.40 |
| **Total Priority 1** | — | **~2.5 h** | **~$2.00** |

> 💡 Compare to ~50+ GPU-h spent on iter9–iter12 against the wrong gate. Priority 1 verdict for $2.

---

## 📋 Implementation order

| Hour | Task | File | Status |
|:-:|:--|:--|:-:|
| 0–1 | `utils/action_labels.py` — derivation, split, JSON writer + self-test | File 1 | ✅ shipped + self-test PASS (9,951 clips) |
| 1 | `utils/vjepa2_imports.py` — add `get_attentive_classifier` + import line | File 2 | ✅ shipped + REPL verified (130 M probe forward OK) |
| 1–3 | `m06d_action_probe.py` labels + features stages + encoder loaders | File 3 | ✅ coded (Stage 1 verified, Stages 2 not yet GPU-exercised) |
| 3–5 | `m06d_action_probe.py` train + paired_delta stages | File 3 | ✅ coded (Stages 3+4 not yet exercised) |
| 5–6 | SANITY end-to-end (200 clips) | File 3 | 🚧 only Stage 1 ran via SANITY; Stages 2–4 deferred to deliberate GPU kickoff |
| 6–7 | `m06d_motion_cos.py` — features (with `--share-features`) + cosine + paired_delta | File 4 | ✅ shipped + algorithmic test PASS + fresh-extract path implemented (no deferral) |
| —   | `utils/frozen_features.py` — extracted shared loaders + producer-consumer extractor | new util | ✅ shipped 2026-04-30 (single source-of-truth for ENCODERS + V-JEPA/DINOv2 loaders) |
| 7–8 | `m06d_future_mse.py` — predictor load + forward + per-clip MSE | File 5 | ✅ shipped + CPU smoke-test PASS (mask_gen instantiates, returns correct tensor shapes); GPU forward unexercised |
| 8–9 | `scripts/run_m06d.sh` thin wrapper (9 stages: labels → 2× features → 2× train → P1 GATE → motion features/cosine/Δ → future_mse forward/Δ) | shell | ✅ shipped 2026-04-30 (~200 LoC, `bash -n` clean). Per CLAUDE.md DELETE PROTECTION: no shell prompts (env `CACHE_POLICY_ALL=1\|2`), `set -uo pipefail` (single-stage failure does NOT abort chain), `SKIP_STAGES=1,2` env supports resume, `ENCODERS=...` env supports debug-one-variant. |
| 9–10 | 🔥 FULL run on `data/eval_10k_local`, ~2.5 h GPU, gate verdict | all | ⬜ ready to launch via `tmux new -s m06d ; ./scripts/run_m06d.sh 2>&1 \| tee logs/run_m06d_v1.log` |

---

## ❓ Open questions

| # | Question | Default | Override |
|:-:|:--|:--|:--|
| OQ1 | 3-class (Path A) or 4-class (Path B) action labels? | 3-class — robust, ≥1,300/class | `--enable-monument-class` enriches via `scene_type=heritage_tourist` |
| OQ2 | Frame count per clip | 16 (V-JEPA 2.1 train default) | `--num-frames 64` (Meta eval — ~4× longer) |
| OQ3 | Train/val/test split | 70 / 15 / 15 stratified, seed=99 | `--seed`, edit `stratified_split` ratios |
| OQ4 | Reuse pooled features between Module 1 and Module 2? | yes via `--share-features` (~30 min × 2 saved) | `--no-share-features` to isolate |
| OQ5 | DINOv2 video recipe — tile + temporal pool, or single mid-frame? | tile + temporal pool (V-JEPA 2 paper §4.1) | `--dinov2-recipe single_frame` |
| OQ6 | Module 3 — predictor checkpoint source | inside V-JEPA 2.1 release `.pt` | `--predictor-ckpt <path>` if separate |

---

## 🗺️ Output directory layout

```text
outputs/full/m06d_action_probe/
├── action_labels.json                 # 70/15/15 splits + class assignments
├── class_counts.json                  # diagnostic: clips/class/split
├── vjepa_2_1_frozen/
│   ├── features_train.npy             # (N_train, n_tokens, D)
│   ├── features_val.npy
│   ├── features_test.npy
│   ├── clip_keys_{train,val,test}.npy # (N,) str object array
│   ├── probe.pt                       # AttentiveClassifier state dict
│   ├── train_log.jsonl                # per-epoch loss/val_acc/lr
│   ├── test_predictions.npy           # (N_test,) ∈ {0,1} per-clip top-1 correctness
│   └── test_metrics.json              # {top1_acc, top1_ci, ...}
├── dinov2/
│   └── ... (same shape)
└── m06d_paired_delta.json             # 🔥 GATE artifact

outputs/full/m06d_motion_cos/
├── vjepa_2_1_frozen/
│   ├── pooled_features_test.npy       # (N_test, D)
│   ├── per_clip_motion_cos.npy        # (N_test,) motion_score per clip
│   └── intra_inter_ratio.json         # {pos_mean, neg_mean, score_mean+ci}
├── dinov2/{...}
└── m06d_motion_cos_paired.json

outputs/full/m06d_future_mse/
├── vjepa_2_1_frozen/
│   ├── per_clip_mse.npy               # (N_test,) float32
│   └── aggregate_mse.json
└── m06d_future_mse_per_variant.json   # priorities 2/3 future-fill; DINOv2 = "n/a"
```

---

## 🔗 Cross-references

- 🔬 Source of the 3 measurements: `iter/iter13_motion_probe_eval/analysis.md` Q2
- 🏷️ Section schema: `src/m00b_fetch_durations.py:43-77`
- 🎲 Subset sampling: `src/m00c_sample_subset.py`
- 📂 Data download: `src/utils/hf_outputs.py:432-435`
- 🧱 Meta `AttentiveClassifier`: `deps/vjepa2/src/models/attentive_pooler.py:103`
- 🔧 Coding contract: `src/CLAUDE.md`
- 📊 Empirical record: `iter/utils/experiment_log.md`
- 🐛 Errors / fixes: `iter/iter13_motion_probe_eval/errors_N_fixes.md`

---

## 📝 Recap (auto-update at end of each work session)

| Date | Author | What changed | Next step |
|:--|:--|:--|:--|
| 2026-04-30 | initial v1 | 3-module priority-1 plan landed | build `utils/action_labels.py` |
| 2026-04-30 | detailed coding plan v2 | function-level signatures + edge cases per file | start coding File 1 |
| 2026-04-30 | shipped v3 | Files 1+2+3 coded, Files 1+2 verified, File 3 Stage 1 verified end-to-end on 9,951 clips | build File 4 (`m06d_motion_cos.py`) |
| 2026-04-30 | shipped v4 | File 4 (motion_cos) + new `utils/frozen_features.py` (extracted shared loaders, no deferral). Algorithmic test PASS on toy clips. Single source-of-truth for ENCODERS verified across 3 sites. | build File 5 (`m06d_future_mse.py`) |
| 2026-04-30 | shipped v5 | File 5 (future_mse) + CPU smoke-test PASS (mask_gen + imports). All 5 files (3 m06d modules + 2 utils) shipped & lint-clean. GPU forward stages await deliberate kickoff. | wire `scripts/run_m06d.sh` thin wrapper, then FULL run |
| 2026-04-30 | shipped v6 | `scripts/run_m06d.sh` 9-stage orchestrator shipped (200 LoC, `bash -n` clean). Env-overridable: `CACHE_POLICY_ALL`, `SKIP_STAGES`, `ENCODERS`, `EVAL_SUBSET`, `LOCAL_DATA`, `ENCODER_CKPT`, `NUM_FRAMES`. **Runbook intentionally omitted** — orchestrator wraps every stage with proper logging. | 🔥 launch FULL run on `data/eval_10k_local` (~2.5 h GPU) for P1 gate verdict |
