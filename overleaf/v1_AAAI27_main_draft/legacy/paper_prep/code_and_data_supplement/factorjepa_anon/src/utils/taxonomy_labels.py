"""Taxonomy label derivation (shared util). CPU-only, pure functions.

iter16 §3.2: extracted from probe_taxonomy.py so the ANNOTATION-band entry-point
(src/m04f_taxonomy_labels.py) and any other consumer single-source the derivation —
no per-module re-implementation (src/CLAUDE.md "SHARED DERIVATION VIA CLI"). The
EVALUATION-band probe (src/m12c_taxonomy.py) CONSUMES the written taxonomy_labels.json;
only the labels entry-point calls derive_taxonomy_labels.

Builds per-clip per-dim ground truth from VLM tags.json + a tag_taxonomy.json spec:
13 single-label + 2 multi-label dims (the legacy path-derived "action" dim was dropped
iter13 v12 — motion-flow lives in utils.action_labels / m04e instead).
"""
import json
from pathlib import Path


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
    # — optical-flow-derived motion class — lives in utils.action_labels / m04e
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


__all__ = ["derive_taxonomy_labels"]
