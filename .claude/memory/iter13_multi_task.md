---
name: iter13 multi-task probe-loss wiring — design + state
description: The iter13 pivot — multi-task supervision on top of JEPA L1, 16 dims, 5 helpers, opt-in per-config
type: project
---

# iter13 multi-task probe-loss supervision — design + state

## Why it exists

Pre-iter13 m09a/m09c trained on pure JEPA L1 + drift L2 only. The downstream gate is a 16-dim probe top-1 / sample-F1 metric. With ZERO gradient signal toward that metric, the probability that a continually-pretrained encoder beats frozen on probe accuracy depends on incidental representation alignment (~5-15% empirically per the iter11/iter12 negative results).

iter13 adds direct gradient signal:
```
total_loss = α · JEPA_L1  +  β · Σ_d w_d · L_d  +  drift_L2
            (α=1.0)        (β=0.1, weight_probe)  (lambda_reg=1.0 for pretrain)
```

| Component | Default | Where |
|---|---|---|
| α (weight_jepa) | 1.0 | `cfg.multi_task_probe.weight_jepa` |
| β (weight_probe) | 0.1 | `cfg.multi_task_probe.weight_probe` |
| w_d (weight_per_dim) | `null` → equal split (1/n_dims) | `cfg.multi_task_probe.weight_per_dim` |
| L_d (per-dim loss) | CrossEntropy (single) / BCEWithLogits (multi) | `compute_multi_task_probe_loss` |
| Head LR multiplier | 10× base_lr | `cfg.multi_task_probe.head_lr_multiplier` |

`weight_per_dim` resolution paths in `compute_multi_task_probe_loss`:
- `null` → equal (1/n_dims) — recommended default
- `int/float` → scalar broadcast
- `dict` → per-dim override (missing dims fall back to 1/n_dims)
- `"kendall_uw"` → NotImplementedError raised (Kendall CVPR 2018 uncertainty weighting; future work)
- other string / type → ValueError / TypeError

## The 16 dims (`probe_taxonomy.py --stage labels`)

| Dim | Type | n_classes | Source |
|---|---|---|---|
| `action` | single | 3 | path-derived (walking/driving/drone) via `utils.action_labels.parse_action_from_clip_key` |
| `scene_type` | single | 13 | `tags.json` via VLM (m04) |
| `time_of_day` | single | 2 | tags.json |
| `weather` | single | 4 | tags.json |
| `lighting` | single | 5 | tags.json |
| `crowd_density` | single | 4 | tags.json |
| `road_layout` | **multi** | 8 | tags.json (multi-hot) |
| `notable_objects` | **multi** | 15 | tags.json (multi-hot) |
| ... 8 more single-label dims from configs/tag_taxonomy.json | single | 2-13 | tags.json |

Total: 14 single-label + 2 multi-label = 16 dims. Total head params ≈ 140K (16 × n_classes × 1664-D + biases).

Output: `outputs/<mode>/probe_taxonomy/taxonomy_labels.json`:
```json
{
  "dims": {"<dim_name>": {"type": "single|multi", "values": [...], "default": ..., "n_classes": N}},
  "labels": {"<clip_key>": {"<dim_name>": <int> | <list[int]>}}
}
```

## The 5 integration helpers (`utils/multi_task_loss.py`)

| Helper | Signature | Purpose |
|---|---|---|
| `merge_multi_task_config(cfg, args, mode_key)` | in-place mutate cfg | Per-mode flatten of `enabled` dict + CLI overrides (`--taxonomy-labels-json`, `--no-multi-task`) |
| `build_multi_task_head_from_cfg(cfg, device)` | → `(head, labels, dims, mt_cfg)` | Construct head + load labels; silent-disable if labels file missing |
| `attach_head_to_optimizer(optimizer, mt_head, mt_cfg, base_lr)` | in-place mutate optimizer | Add head's params as separate group at `base_lr × head_lr_multiplier` |
| `run_multi_task_step(student, mt_head, mt_cfg, mt_labels, mt_dims, batch_clips, batch_keys, scaler, mp_cfg, dtype, device)` | → `(mt_loss_val, mt_per_dim)` | One forward+backward; re-raises `OutOfMemoryError` for caller's per-loop policy |
| `export_multi_task_head(mt_head, dims_spec, d_encoder, path)` | side effect | Write `multi_task_head.pt` next to student_encoder.pt |

Each m09 call site shrinks from ~22 LoC to ~3 LoC. Net 120 duplicated LoC → 30 LoC across both files + 95 LoC of helpers (single test surface).

All 5 helpers are no-op when `mt_head is None`, so the m09 call sites never need a `if mt_head is not None:` guard around them.

## Per-config opt-in pattern

`base_optimization.yaml` ships:
```yaml
multi_task_probe:
  enabled:                  {sanity: false, poc: false, full: false}    # opt-in per config
  labels_path:              outputs/full/probe_taxonomy/taxonomy_labels.json
  weight_jepa:              1.0
  weight_probe:             0.1
  weight_per_dim:           null
  head_lr_multiplier:       10.0
```

Three iter13 leaf configs override:
```yaml
# probe_pretrain.yaml + surgery_3stage_DI.yaml + surgery_2stage_noDI.yaml
multi_task_probe:
  enabled: {sanity: true, poc: true, full: true}    # iter13 opt-in
```

Legacy `configs/legacy2/ch10_pretrain.yaml` and `configs/legacy2/explora.yaml` are unaffected (no override → inherit base default `false` for all modes).

## Train-loop integration sequence (m09a/m09c)

```python
# 1. Config flatten (in merge_config_with_args)
merge_multi_task_config(cfg, args, mode_key)

# 2. Head construction (in train(), after grad-ckpt enable, before optimizer build)
mt_head, mt_labels_by_clip, mt_dims_spec, mt_cfg = build_multi_task_head_from_cfg(cfg, device)

# 3. Optimizer extension (after build_optimizer)
attach_head_to_optimizer(optimizer, mt_head, mt_cfg, cfg["optimization"]["lr"])
# m09c also re-attaches at every per-stage optimizer rebuild (since optim is re-created each stage)

# 4. Per-step forward+backward (after _train_step_grad_accum returns successfully, before scaler.unscale_)
try:
    mt_loss_val, mt_per_dim = run_multi_task_step(
        student, mt_head, mt_cfg, mt_labels_by_clip, mt_dims_spec,
        batch_clips, batch_keys, scaler, mp_cfg, dtype, device)
except torch.cuda.OutOfMemoryError:
    optimizer.zero_grad(); torch.cuda.empty_cache(); print("OOM in multi-task fwd"); continue

# 5. Grad clip extension (optional but recommended — include head params)
_clip_params = list(student.parameters()) + list(predictor.parameters())
if mt_head is not None: _clip_params += list(mt_head.parameters())
grad_norm = torch.nn.utils.clip_grad_norm_(_clip_params, cfg["optimization"]["grad_clip"])

# 6. Logging (per-step JSONL + wandb)
if mt_head is not None:
    step_record["loss_multi_task"] = round(mt_loss_val, 6)
    step_record["loss_multi_task_per_dim"] = {d: round(v, 6) for d, v in mt_per_dim.items()}
    wb_metrics["loss/multi_task"] = mt_loss_val
    for d, v in mt_per_dim.items(): wb_metrics[f"loss/multi_task/{d}"] = v

# 7. Export (after final student_encoder.pt export)
export_multi_task_head(mt_head, mt_dims_spec, cfg["model"]["embed_dim"],
                      output_dir / "multi_task_head.pt")
```

## Data path: how clip_keys reach `run_multi_task_step`

`compute_multi_task_probe_loss` needs `clip_keys` (a list of strings) aligned with the batch dim of pooled features so it can look up labels in `labels_by_clip`.

- **m09a** (continual SSL via producer thread): `producer_thread` was modified in iter13 to yield `(msg_type, batch_clips, batch_keys)` instead of `(msg_type, batch_clips)`. The main thread unpacks the 3-tuple at the queue.get().
- **m09c** (factor surgery via streaming OR sampler):
  - Streaming (`StreamingFactorDataset`): yields dicts `{"tensor", "factor_type", "clip_key"}`; collator turns clip_key into a list. Threaded as `batch_keys = list(batch["clip_key"])`.
  - Legacy sampler (`FactorSampler.sample()`): returns `(factor, clip_key, path)` — capture the second element. Threaded as `batch_keys.append(clip_key)`.

If `batch_keys` is empty (shouldn't happen post-iter13, but defensive), `run_multi_task_step` short-circuits to `(0.0, {})`.

## Multi-task forward cost

Adds ONE extra full-batch unmasked encoder forward per macro batch (vs the masked context+target encoder forwards in JEPA step). The forward uses ALL 4608 patch tokens (vs ~50% mask in JEPA), so activation memory is roughly 2× a JEPA encoder forward. With grad-ckpt + bf16, this is ~6-8 GB on a single V-JEPA ViT-G forward at 16-frame full-batch.

That's why this path is fine on 96 GB but stacks badly with the OOM regime on 24 GB. See `hardware_split.md`.

## Validation status

| Test | Outcome | Notes |
|---|---|---|
| 3-check (`py_compile` + `ast.parse` + `ruff F,E9`) on all 3 modified .py | ✅ clean | `m09a_pretrain.py`, `m09c_surgery.py`, `utils/multi_task_loss.py` |
| 11-test REPL smoke on `multi_task_loss.py` | ✅ clean | head construction / loss compute / mixed-validity batch / weight_per_dim variants / kendall_uw raises / optimizer step moves weights / save-load roundtrip / load_taxonomy_labels |
| 11-test REPL smoke on the 5 integration helpers | ✅ clean | per-mode flatten / CLI override / no-op when block missing / disabled cfg / missing labels file → silent disable / optimizer attach / head fwd+bwd / no-op short-circuits / save-load roundtrip |
| End-to-end SANITY pretrain (24 GB) | ❌ OOM | Hardware-bound, NOT a wiring bug. Move to 96 GB |
| End-to-end SANITY pretrain (96 GB) | ⏳ NOT YET RUN | Will exercise multi-task forward on real data for the first time |
| End-to-end FULL pretrain | ⏳ NOT YET RUN | Open empirical question: does multi-task move probe top-1 above frozen? |

## What's NOT yet implemented

- **`weight_per_dim: "kendall_uw"`** — Kendall CVPR 2018 learnable log-σ weighting. Raises NotImplementedError. Would need a separate `nn.Module` owning `log_sigma_d` per dim + optimizer wiring. iter12 v3 attempted similar `UncertaintyWeights` for InfoNCE/TCC mix; that prior art is in `utils.training.UncertaintyWeights`. Out of scope for v1.
- **Sample-F1 evaluation of multi-label dims at probe time** — `probe_taxonomy.py --stage train` computes per-clip F1 for the 2 multi-label dims, but the train-time multi-task loss uses BCEWithLogits (not F1 directly). That's standard — F1 is non-differentiable; BCE is the proxy.
- **mt_head warm-start at eval time** — `multi_task_head.pt` is exported but `probe_taxonomy.py --stage train` doesn't currently load it as initialization. Could be a small lift for the eval-side probe accuracy.
