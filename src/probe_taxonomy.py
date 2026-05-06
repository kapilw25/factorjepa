"""Unified per-dimension probe — top-1 accuracy (single-label) +
sample-F1 (multi-label) across **15 dims**: 13 single-label + 2 multi-label
from configs/tag_taxonomy.json. The legacy 16th dim (path-derived 3-class
action) was DROPPED in iter13 v12 (2026-05-05) — its replacement is the
optical-flow-derived motion class in probe_action.py (Phase 2 of
plan_code_dev.md), which is no longer single-frame-solvable retrieval.

Multi-label dims (road_layout, notable_objects) use per-clip F1 (sample-F1)
— bootstrap-friendly. Single-label dims (action + 13 from taxonomy) use top-1.

Reuses probe_action's cached features under <features-root>/<encoder>/
(features_{train,val,test}.npy + clip_keys_{train,val,test}.npy). Probe head
is the same Meta AttentiveClassifier from utils.vjepa2_imports.

Stages:
  labels        — derive per-clip per-dim ground truth from tags.json +
                  tag_taxonomy.json (CPU, ~30 s)
  train         — per-encoder × per-dim probe head training (GPU, ~5 min)
  paired_delta  — N-way per-dim Δ across encoders + corpus-level summary (CPU)
  plot          — per-dim grouped bar chart (CPU, ~5 s)

USAGE (orchestrated by scripts/run_probe_eval.sh as Stages 11-14, optional):
    python -u src/probe_taxonomy.py --SANITY --stage labels \\
        --tags-json data/eval_10k_local/tags.json \\
        --tag-taxonomy configs/tag_taxonomy.json \\
        --eval-subset data/eval_10k_sanity.json \\
        --output-root outputs/sanity/probe_taxonomy --cache-policy 1

    python -u src/probe_taxonomy.py --SANITY --stage train \\
        --encoder vjepa_2_1_frozen \\
        --features-root outputs/sanity/probe_action \\
        --output-root outputs/sanity/probe_taxonomy --cache-policy 1

    python -u src/probe_taxonomy.py --SANITY --stage paired_delta \\
        --output-root outputs/sanity/probe_taxonomy --cache-policy 1

    python -u src/probe_taxonomy.py --SANITY --stage plot \\
        --output-root outputs/sanity/probe_taxonomy --cache-policy 1
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from utils.bootstrap import bootstrap_ci, paired_bca
from utils.cache_policy import (
    add_cache_policy_arg, guarded_delete, resolve_cache_policy_interactive,
)
from utils.checkpoint import save_json_checkpoint
from utils.config import check_gpu
from utils.gpu_batch import cleanup_temp
from utils.plots import COLORS, ENCODER_COLORS, init_style, save_fig
from utils.progress import make_pbar
from utils.vjepa2_imports import get_attentive_classifier
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics


NUM_FRAMES_DEFAULT = 16


# ── Label derivation ─────────────────────────────────────────────────

def _filter_dims(taxonomy: dict) -> dict:
    """Drop _comment / _source / _changelog meta fields; return real dims only."""
    return {k: v for k, v in taxonomy.items() if not k.startswith("_")}


def derive_taxonomy_labels(tags_json: Path, tag_taxonomy: Path,
                            eval_subset: Path) -> tuple:
    """Build per-clip per-dim labels from tags.json + taxonomy.

    Returns (labels_by_clip, dims_spec):
      labels_by_clip: {clip_key: {dim_name: int (single) | list[int] (multi-hot)}}
      dims_spec:      {dim_name: {type, values, n_classes, default}}
    """
    taxonomy = json.loads(tag_taxonomy.read_text())
    dims_raw = _filter_dims(taxonomy)
    dims = {}
    # iter13 v12 (2026-05-05): "action" dim DROPPED here. The path-derived
    # 3-class (walking/driving/drone) was retrieval, not motion. Its replacement
    # — optical-flow-derived motion class — lives in probe_action.py
    # (action_labels.json from m04d motion_features.npy). Taxonomy now reports
    # 15 dims: 13 single-label + 2 multi-label, all from tag_taxonomy.json.
    for name, spec in dims_raw.items():
        dims[name] = {
            "type":      spec["type"],          # 'single' or 'multi'
            "values":    spec["values"],
            "default":   spec["default"],
            "n_classes": len(spec["values"]),
        }

    tags_list = json.loads(tags_json.read_text())
    # Tags keyed by `source_file` (basename). Map clip_key → tag dict via
    # Path(clip_key).name == tag["source_file"].
    tag_by_basename = {t["source_file"]: t for t in tags_list if "source_file" in t}

    eval_keys = json.loads(eval_subset.read_text())["clip_keys"]

    labels_by_clip = {}
    skipped_no_tag = 0
    for k in eval_keys:
        basename = Path(k).name
        if basename not in tag_by_basename:
            skipped_no_tag += 1
            continue
        t = tag_by_basename[basename]
        per_dim = {}
        # iter13 v12 (2026-05-05): action dim removed here. All 15 dims come
        # from tag_taxonomy.json now.
        for dim_name, spec in dims.items():
            v = t.get(dim_name, spec["default"])
            if spec["type"] == "single":
                # Coerce VLM-out-of-vocab values to default.
                if v not in spec["values"]:
                    v = spec["default"]
                per_dim[dim_name] = spec["values"].index(v)
            else:                                          # multi-label
                if isinstance(v, str):
                    v = [v]
                # Multi-hot vector — index of each value in spec.values, OR
                # silently drop unknown tag values (VLM can hallucinate).
                multi_hot = [0] * spec["n_classes"]
                for tag_v in v:
                    if tag_v in spec["values"]:
                        multi_hot[spec["values"].index(tag_v)] = 1
                per_dim[dim_name] = multi_hot
        labels_by_clip[k] = per_dim

    print(f"  derived labels for {len(labels_by_clip)}/{len(eval_keys)} clips "
          f"({skipped_no_tag} skipped — no tag record)")
    return labels_by_clip, dims


def run_labels_stage(args, wb) -> None:
    if any(p is None for p in (args.tags_json, args.tag_taxonomy, args.eval_subset)):
        sys.exit("FATAL: --stage labels requires --tags-json + --tag-taxonomy + --eval-subset")
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / "taxonomy_labels.json"
    if out_path.exists() and args.cache_policy == "1":
        print(f"  [keep] {out_path} present — skipping (--cache-policy 2 to redo)")
        return
    guarded_delete(out_path, args.cache_policy, "taxonomy_labels.json")

    labels_by_clip, dims = derive_taxonomy_labels(
        args.tags_json, args.tag_taxonomy, args.eval_subset)
    out = {"dims": dims, "labels": labels_by_clip}
    save_json_checkpoint(out, out_path)
    print(f"Wrote: {out_path}  ({len(dims)} dims, {len(labels_by_clip)} labeled clips)")
    log_metrics(wb, {"n_clips_labeled": len(labels_by_clip), "n_dims": len(dims)})


# ── Probe head ────────────────────────────────────────────────────────

def _make_probe(d_in: int, n_classes: int, depth: int = 4):
    AC = get_attentive_classifier()
    return AC(embed_dim=d_in, num_classes=n_classes, depth=depth, num_heads=16,
              mlp_ratio=4.0, complete_block=True, use_activation_checkpointing=False)


def _train_one_dim(probe, X_tr, y_tr, X_val, y_val, dim_type: str,
                    epochs: int, lr: float, wd: float, batch_size: int,
                    warmup_pct: float, seed: int):
    """Train probe head for one dim. Returns best probe state by val metric.

    For single-label: y is (N,) int → CrossEntropyLoss → top-1 acc by val
    For multi-label:  y is (N, C) multi-hot float → BCEWithLogitsLoss → mean-per-clip F1 by val
    """
    device = "cuda"
    probe = probe.to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    n_tr = len(y_tr)
    bs = max(8, min(batch_size, n_tr))
    steps_per_epoch = math.ceil(n_tr / bs)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(0, int(total_steps * warmup_pct))

    def lr_lambda(s):
        if warmup_steps > 0 and s < warmup_steps:
            return s / warmup_steps
        progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    rng = np.random.default_rng(seed)
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = (torch.from_numpy(y_tr).long() if dim_type == "single"
              else torch.from_numpy(y_tr).float())
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = (torch.from_numpy(y_val).long() if dim_type == "single"
               else torch.from_numpy(y_val).float())

    best_val_metric = -1.0
    best_state = None
    step = 0
    for epoch in range(epochs):
        probe.train()
        idx = rng.permutation(n_tr)
        for s in range(0, n_tr, bs):
            sub = idx[s:s + bs]
            xb = X_tr_t[sub].to(device, non_blocking=True)
            yb = y_tr_t[sub].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = probe(xb)
            if dim_type == "single":
                loss = F.cross_entropy(logits, yb)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optim.step()
            sched.step()
            step += 1

        # Val eval
        probe.eval()
        with torch.no_grad():
            v_logits = probe(X_val_t.to(device)).cpu()
            if dim_type == "single":
                metric = float((v_logits.argmax(-1) == y_val_t).float().mean())
            else:
                pred = (v_logits > 0).float()
                metric = float(_per_clip_f1(pred.numpy(), y_val_t.numpy()).mean())
        if metric > best_val_metric:
            best_val_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
    return best_val_metric, best_state


def _per_clip_f1(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-clip F1 on multi-label predictions. Both pred + gt are (N, C) {0,1}.
    Returns (N,) float — F1 per clip. F1=0 when prediction is empty AND gt is empty
    (an arbitrary convention; alternative is F1=1 in that degenerate case).
    """
    tp = (pred * gt).sum(axis=1).astype(np.float64)
    fp = (pred * (1 - gt)).sum(axis=1).astype(np.float64)
    fn = ((1 - pred) * gt).sum(axis=1).astype(np.float64)
    denom = 2 * tp + fp + fn
    # np.divide with where= mask avoids division-by-zero warnings on empty cases.
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=(denom > 0))


def _eval_test(probe, X_test, y_test, dim_type: str):
    """Per-clip metric vector + scalar mean + BCa CI."""
    probe.eval()
    with torch.no_grad():
        logits = probe(torch.from_numpy(X_test).float().to("cuda")).cpu()
    if dim_type == "single":
        per_clip = (logits.argmax(-1).numpy() == y_test).astype(np.float64)
        metric_name = "top1_acc"
    else:
        pred = (logits > 0).float().numpy()
        per_clip = _per_clip_f1(pred, y_test).astype(np.float64)
        metric_name = "sample_f1"
    mean_v = float(per_clip.mean())
    ci = bootstrap_ci(per_clip)
    return per_clip, mean_v, ci, metric_name


# ── Stage: train ──────────────────────────────────────────────────────

def run_train_stage(args, wb) -> None:
    if args.encoder is None:
        sys.exit("FATAL: --stage train requires --encoder")
    if args.features_root is None:
        sys.exit("FATAL: --stage train requires --features-root (probe_action's per-encoder cache)")
    check_gpu()
    cleanup_temp()

    labels_path = args.output_root / "taxonomy_labels.json"
    if not labels_path.exists():
        sys.exit(f"FATAL: {labels_path} not found — run --stage labels first")
    tx = json.loads(labels_path.read_text())
    dims, labels_by_clip = tx["dims"], tx["labels"]

    enc_dir = args.features_root / args.encoder

    # iter13 lazy-extract (2026-05-05): mirrors probe_action.run_train_stage —
    # if features_<split>.npy is missing on disk (the new default per the
    # eval-disk-budget refactor), extract in-memory via extract_features_for_keys.
    # The function's resume side-effect (.probe_features_<split>_ckpt.npz) is
    # SHARED with probe_action's lazy cache, so when probe_taxonomy runs
    # AFTER probe_action --stage train (per scripts/run_probe_eval.sh's
    # per-encoder loop), the cache is already populated → near-zero re-extract
    # cost. Encoder is loaded only when at least one split needs extraction.
    from utils.frozen_features import (
        ENCODERS, extract_features_for_keys,
        load_dinov2_frozen, load_vjepa_2_1_frozen,
    )
    from utils.action_labels import load_action_labels
    from utils.data_download import ensure_local_data

    enc_kind = ENCODERS[args.encoder]["kind"]
    needs_extract = [s for s in ("train", "val", "test")
                     if not (enc_dir / f"features_{s}.npy").exists()]
    model = crop = embed_dim = None
    if needs_extract:
        if (enc_kind == "vjepa" and args.encoder_ckpt is None) or args.local_data is None:
            sys.exit(
                f"FATAL: probe_taxonomy --stage train needs to lazily extract "
                f"{needs_extract} features but --encoder-ckpt and/or --local-data "
                f"are missing. Pass them so the encoder can be loaded.\n"
                f"  Alternatively, re-run probe_action with "
                f"`--features-splits train val test` to materialise the .npy files."
            )
        print(f"  [lazy-extract] missing splits on disk: {needs_extract} — "
              f"loading {args.encoder} encoder for in-memory extraction")
        ensure_local_data(args)
        if enc_kind == "vjepa":
            model, crop, embed_dim = load_vjepa_2_1_frozen(args.encoder_ckpt, args.num_frames)
        elif enc_kind == "dinov2":
            model, _proc, crop, embed_dim = load_dinov2_frozen()
        else:
            sys.exit(f"FATAL: unknown encoder kind '{enc_kind}'")

    # Group clip_keys by split (deterministic order from action_labels.json,
    # which is the source-of-truth for the 70/15/15 split per iter13 plan).
    action_labels_path = args.features_root / "action_labels.json"
    action_labels = load_action_labels(action_labels_path)
    by_split = {"train": [], "val": [], "test": []}
    for k, info in action_labels.items():
        by_split[info["split"]].append(k)

    def _load_or_extract(split: str):
        npy_path  = enc_dir / f"features_{split}.npy"
        keys_path = enc_dir / f"clip_keys_{split}.npy"
        if npy_path.exists() and keys_path.exists():
            return np.load(npy_path), [str(k) for k in np.load(keys_path, allow_pickle=True)]
        feats, ordered_keys = extract_features_for_keys(
            args, model, enc_kind, crop, embed_dim,
            by_split[split], enc_dir, label=f"features_{split}",
            pool_tokens=(args.pool_tokens if args.pool_tokens > 0 else None),
        )
        return feats, [str(k) for k in ordered_keys]

    feats_train, keys_train = _load_or_extract("train")
    feats_val,   keys_val   = _load_or_extract("val")
    feats_test,  keys_test  = _load_or_extract("test")

    # iter13 (2026-05-05): free encoder GPU memory BEFORE per-dim probe-head
    # training. Heads are tiny (~140K params each, 16 dims) but they still
    # compete with the dangling 7.5 GB encoder for cuda allocator blocks.
    # gc.collect() is required (see probe_action.py:run_train_stage comment).
    if model is not None:
        del model
        import gc
        gc.collect()
        import torch as _torch
        _torch.cuda.empty_cache()
        _torch.cuda.ipc_collect()

    out_dir = args.output_root / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"encoder": args.encoder, "dims": {}}
    pbar = make_pbar(total=len(dims), desc=f"probe_taxonomy[{args.encoder}]", unit="dim")

    for dim_name, spec in dims.items():
        # Filter to clips that have a label for this dim (most clips will).
        def _idx(keys):
            return [i for i, k in enumerate(keys) if k in labels_by_clip and dim_name in labels_by_clip[k]]
        ti = _idx(keys_train); vi = _idx(keys_val); te = _idx(keys_test)
        if len(ti) < 5 or len(vi) < 2 or len(te) < 5:
            print(f"  [skip] dim={dim_name}: too few labeled clips (train={len(ti)}, val={len(vi)}, test={len(te)})")
            summary["dims"][dim_name] = {"skipped": True, "reason": "too_few_clips"}
            pbar.update(1)
            continue

        if spec["type"] == "single":
            y_tr = np.array([labels_by_clip[keys_train[i]][dim_name] for i in ti], dtype=np.int64)
            y_val = np.array([labels_by_clip[keys_val[i]][dim_name] for i in vi], dtype=np.int64)
            y_te = np.array([labels_by_clip[keys_test[i]][dim_name] for i in te], dtype=np.int64)
        else:
            y_tr = np.array([labels_by_clip[keys_train[i]][dim_name] for i in ti], dtype=np.float32)
            y_val = np.array([labels_by_clip[keys_val[i]][dim_name] for i in vi], dtype=np.float32)
            y_te = np.array([labels_by_clip[keys_test[i]][dim_name] for i in te], dtype=np.float32)

        d_in = int(feats_train.shape[-1])
        probe = _make_probe(d_in, spec["n_classes"], depth=args.probe_depth)
        best_val, best_state = _train_one_dim(
            probe, feats_train[ti], y_tr, feats_val[vi], y_val,
            dim_type=spec["type"], epochs=args.epochs, lr=args.probe_lr,
            wd=args.probe_wd, batch_size=args.train_batch_size,
            warmup_pct=args.warmup_pct, seed=args.seed)
        probe.load_state_dict(best_state)
        per_clip, mean_v, ci, metric_name = _eval_test(probe, feats_test[te], y_te, spec["type"])
        # Persist per-dim outputs.
        torch.save(best_state, out_dir / f"probe_{dim_name}.pt")
        np.save(out_dir / f"per_clip_{dim_name}.npy", per_clip)
        np.save(out_dir / f"clip_keys_test_{dim_name}.npy",
                np.array([keys_test[i] for i in te], dtype=object))
        dim_record = {
            "type": spec["type"], "n_classes": spec["n_classes"],
            "metric": metric_name,
            "n_train": len(ti), "n_val": len(vi), "n_test": len(te),
            "best_val": round(float(best_val), 6),
            "test_mean": round(mean_v, 6),
            "test_ci": ci,
        }
        summary["dims"][dim_name] = dim_record
        log_metrics(wb, {f"{args.encoder}/{dim_name}/{metric_name}": mean_v})
        pbar.update(1)
    pbar.close()
    save_json_checkpoint(summary, out_dir / "test_metrics.json")
    print(f"Wrote: {out_dir / 'test_metrics.json'} ({len(summary['dims'])} dim records)")


# ── Stage: paired_delta (N-way per-dim) ──────────────────────────────

def run_paired_delta_stage(args, wb) -> None:
    """N-way per-dim Δ across encoders. Schema:
        {"dims": {<dim>: {type, metric, n_classes,
                          by_encoder: {<enc>: {test_mean, test_ci, n_test}, ...},
                          pairwise_deltas: {"<a>_minus_<b>": {delta, ci_lo, ci_hi, p_value, gate_pass}, ...}
                         }, ...}}
    """
    enc_dirs = sorted([p for p in args.output_root.iterdir()
                       if p.is_dir() and (p / "test_metrics.json").exists()])
    if len(enc_dirs) < 2:
        sys.exit(f"FATAL: need >=2 encoders with test_metrics.json, found: {[p.name for p in enc_dirs]}")
    print(f"Encoders found: {[p.name for p in enc_dirs]}")

    enc_data = {}
    for ed in enc_dirs:
        enc_data[ed.name] = json.loads((ed / "test_metrics.json").read_text())["dims"]

    # Union of dim names across encoders. Any dim missing in any encoder → skipped.
    common_dims = set.intersection(*[set(d.keys()) for d in enc_data.values()])
    out = {"dims": {}}
    for dim_name in sorted(common_dims):
        if any(enc_data[e][dim_name].get("skipped") for e in enc_data):
            continue
        sample = next(iter(enc_data.values()))[dim_name]
        by_encoder = {e: {"test_mean": enc_data[e][dim_name]["test_mean"],
                          "test_ci":   enc_data[e][dim_name]["test_ci"],
                          "n_test":    enc_data[e][dim_name]["n_test"]}
                      for e in enc_data}
        # Pairwise paired-Δ requires loading per-clip arrays + key-aligning.
        pairwise = {}
        names = sorted(enc_data.keys())
        per_clip_cache = {}
        keys_cache = {}
        for e in names:
            ed = args.output_root / e
            per_clip_cache[e] = np.load(ed / f"per_clip_{dim_name}.npy")
            keys_cache[e] = [str(k) for k in np.load(ed / f"clip_keys_test_{dim_name}.npy",
                                                     allow_pickle=True)]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ka, kb = keys_cache[a], keys_cache[b]
                shared = sorted(set(ka) & set(kb))
                if not shared:
                    continue
                ai = {k: idx for idx, k in enumerate(ka)}
                bi = {k: idx for idx, k in enumerate(kb)}
                a_arr = np.array([per_clip_cache[a][ai[k]] for k in shared], dtype=np.float64)
                b_arr = np.array([per_clip_cache[b][bi[k]] for k in shared], dtype=np.float64)
                delta = a_arr - b_arr
                bca = paired_bca(delta)
                pairwise[f"{a}_minus_{b}"] = {
                    "n_shared":     int(len(shared)),
                    "delta":        round(float(delta.mean()), 6),
                    "ci_lo":        round(float(bca["ci_lo"]), 6),
                    "ci_hi":        round(float(bca["ci_hi"]), 6),
                    "ci_half":      round(float(bca["ci_half"]), 6),
                    "p_value":      float(bca["p_value_vs_zero"]),
                    "gate_pass":    bool(bca["ci_lo"] > 0),
                }
        out["dims"][dim_name] = {
            "type":             sample["type"],
            "metric":           sample["metric"],
            "n_classes":        sample["n_classes"],
            "by_encoder":       by_encoder,
            "pairwise_deltas":  pairwise,
        }

    out_path = args.output_root / "per_dim_acc.json"
    save_json_checkpoint(out, out_path)
    log_metrics(wb, {"n_encoders": len(enc_data), "n_dims_compared": len(out["dims"])})
    print(f"Wrote: {out_path}  ({len(out['dims'])} dims compared)")


# ── Stage: plot ───────────────────────────────────────────────────────

def run_plot_stage(args, wb) -> None:
    src = args.output_root / "per_dim_acc.json"
    if not src.exists():
        sys.exit(f"FATAL: {src} not found — run --stage paired_delta first")
    data = json.loads(src.read_text())
    dims = data["dims"]
    if not dims:
        print("  no dims to plot"); return

    init_style()
    encoders = sorted(set(e for d in dims.values() for e in d["by_encoder"]))
    n_enc = len(encoders)
    dim_names = sorted(dims.keys())
    n_dims = len(dim_names)

    # Grid: rows = ceil(n_dims / 3), cols = 3 (each cell is one dim's grouped bars).
    cols = 3
    rows = math.ceil(n_dims / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    fallback_colors = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["orange"],
                       COLORS["purple"], COLORS["cyan"]]
    enc_color = {e: ENCODER_COLORS.get(e, ENCODER_COLORS.get("vjepa") if e.startswith("vjepa")
                                       else fallback_colors[i % len(fallback_colors)])
                 for i, e in enumerate(encoders)}

    for ax_idx, dim_name in enumerate(dim_names):
        ax = axes[ax_idx]
        d = dims[dim_name]
        means = [d["by_encoder"][e]["test_mean"] for e in encoders]
        errs  = [d["by_encoder"][e]["test_ci"]["ci_half"] for e in encoders]
        x = np.arange(n_enc)
        ax.bar(x, means, 0.6, color=[enc_color[e] for e in encoders],
               yerr=errs, capsize=4, alpha=0.85,
               error_kw={"lw": 1.0, "ecolor": "#222"})
        # Y-limits — auto-pad
        if means:
            arr_v = np.array(means); arr_e = np.array(errs)
            lo = max(0.0, float((arr_v - arr_e).min()))
            hi = min(1.05, float((arr_v + arr_e).max()))
            pad = max(0.10 * (hi - lo), 0.02 * hi) if hi > 0 else 0.05
            ax.set_ylim(max(0.0, lo - pad), hi + pad)
            for xi, m in zip(x, means):
                ax.text(xi, m + pad * 0.3, f"{m:.2f}", ha="center", va="bottom",
                        fontsize=8, color="#222")
        ax.set_xticks(x)
        ax.set_xticklabels([e.replace("vjepa_2_1_", "v.").replace("_", "\n") for e in encoders],
                           fontsize=8)
        ax.set_title(f"{dim_name}\n({d['metric']}, n={d['n_classes']}cls)",
                     fontsize=10, fontweight="bold")
    # Hide spare axes
    for j in range(n_dims, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"probe_taxonomy — per-dim metric across {n_enc} encoders · 95 % BCa CI\n"
                 f"(top-1 for single-label, sample-F1 for multi-label)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, str(args.output_root / "probe_taxonomy_per_dim"))


# ── CLI ───────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Per-dimension taxonomy probe (single + multi-label across 15 tag dims; "
                    "action dim dropped iter13 v12 in favor of optical-flow probe_action).")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--stage", required=True,
                   choices=["labels", "train", "paired_delta", "plot"])
    p.add_argument("--encoder", type=str, default=None,
                   help="Encoder name (must match a probe_encoders.yaml key + features-root subdir)")
    p.add_argument("--features-root", type=Path, default=None,
                   help="probe_action output root (per-encoder features_*.npy + clip_keys_*.npy)")
    p.add_argument("--tags-json", type=Path, default=None)
    p.add_argument("--tag-taxonomy", type=Path, default=None)
    p.add_argument("--eval-subset", type=Path, default=None)
    # iter13 (2026-05-05): args needed for lazy-extract path (Stage 2 default
    # only saves features_test.npy now; Stage 3 train extracts train+val
    # in-memory). When all 3 splits are already on disk, these are unused.
    p.add_argument("--encoder-ckpt", type=Path, default=None,
                   help="V-JEPA encoder ckpt (required when train/val features need lazy extraction)")
    from utils.config import add_local_data_arg as _add_local_data_arg
    _add_local_data_arg(p)
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES_DEFAULT,
                   help="Frames per clip for lazy extraction; ignored when features already on disk")
    p.add_argument("--pool-tokens", type=int, default=16,
                   help="Adaptive-avg-pool encoder output to N tokens before storage / probe. "
                        "Default 16 (V-JEPA paper §4 attentive-probe regime). "
                        "Use 0 to disable pooling.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--probe-lr", type=float, default=5e-4)
    p.add_argument("--probe-wd", type=float, default=0.05)
    p.add_argument("--warmup-pct", type=float, default=0.10)
    p.add_argument("--probe-depth", type=int, default=4)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=99)
    add_cache_policy_arg(p)
    add_wandb_args(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        sys.exit("ERROR: specify --SANITY, --POC, or --FULL")
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)
    args.output_root.mkdir(parents=True, exist_ok=True)
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb = init_wandb(f"probe_taxonomy_{args.stage}", mode,
                    config=vars(args), enabled=not args.no_wandb)
    try:
        if args.stage == "labels":
            run_labels_stage(args, wb)
        elif args.stage == "train":
            run_train_stage(args, wb)
        elif args.stage == "paired_delta":
            run_paired_delta_stage(args, wb)
        elif args.stage == "plot":
            run_plot_stage(args, wb)
    finally:
        finish_wandb(wb)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        print(f"\n❌ FATAL: {Path(__file__).name} crashed — see traceback below", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
