"""
Compute 9 evaluation metrics via FAISS-GPU in Easy/Hard mode.
GPU-only (FAISS-GPU required). Reads embeddings{sfx}.npy + tags.json.
Plots saved with encoder suffix to avoid overwrite across encoders.

USAGE:
    python -u src/m06_faiss_metrics.py --encoder vjepa --SANITY 2>&1 | tee logs/m06_vjepa_sanity.log
    python -u src/m06_faiss_metrics.py --encoder vjepa --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_vjepa_poc.log
    python -u src/m06_faiss_metrics.py --encoder dinov2 --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_dinov2_poc.log
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    EMBEDDINGS_FILE, TAGS_FILE, METRICS_FILE, OUTPUTS_DIR,
    FAISS_K_NEIGHBORS, TAG_TAXONOMY_JSON,
    check_gpu, add_subset_arg, get_output_dir,
    add_encoder_arg, get_encoder_files, get_encoder_info, ENCODER_REGISTRY,
)
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, log_image, log_artifact, finish_wandb,
)
from utils.bootstrap import (
    bootstrap_ci, per_clip_prec_at_k, per_clip_map_at_k,
    per_clip_cycle_at_k, per_clip_ndcg_at_k,
)

try:
    import faiss
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install faiss-gpu")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────
EXCLUSION_WINDOW_SEC = 30
DEFAULT_CLIP_DURATION_SEC = 10
CONFIDENCE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SLICE_FIELDS = ["time_of_day", "weather", "crowd_density", "traffic_density",
                "road_surface", "infrastructure_quality", "vegetation", "lighting"]


# ── True Overlap@K (from augmented embeddings) ──────────────────────────

def compute_true_overlap_at_k(output_dir: Path, k: int) -> float:
    """True Overlap@K: kNN IoU between two augmented views of the same clips.

    Loads overlap_augA.npy and overlap_augB.npy (from m05c_true_overlap.py),
    builds FAISS index on each, computes IoU of kNN neighborhoods.
    Returns percentage [0, 100] or None if augmented files missing.
    """
    aug_a_file = output_dir / "overlap_augA.npy"
    aug_b_file = output_dir / "overlap_augB.npy"

    if not aug_a_file.exists() or not aug_b_file.exists():
        return None

    aug_a = np.load(aug_a_file).astype(np.float32)
    aug_b = np.load(aug_b_file).astype(np.float32)

    if aug_a.shape != aug_b.shape:
        print(f"WARNING: augA shape {aug_a.shape} != augB shape {aug_b.shape}")
        return None

    n, d = aug_a.shape
    res = faiss.StandardGpuResources()

    # Build index on view A, find kNN for each clip in A-space
    idx_a = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))
    idx_a.add(aug_a)
    _, I_a = idx_a.search(aug_a, k + 1)  # each clip's neighbors in view A

    # Build index on view B, find kNN for each clip in B-space
    idx_b = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))
    idx_b.add(aug_b)
    _, I_b = idx_b.search(aug_b, k + 1)  # each clip's neighbors in view B

    # Compute per-clip IoU of kNN sets (skip self at col 0)
    per_clip_iou = np.zeros(n)
    for i in range(n):
        set_a = set(I_a[i, 1:k + 1].tolist()) - {-1}
        set_b = set(I_b[i, 1:k + 1].tolist()) - {-1}
        union = set_a | set_b
        if union:
            per_clip_iou[i] = len(set_a & set_b) / len(union)

    from utils.bootstrap import bootstrap_ci
    ci = bootstrap_ci(per_clip_iou * 100)  # percentage scale
    return float(np.mean(per_clip_iou)) * 100, ci


# ── Load taxonomy ────────────────────────────────────────────────────────

def load_taxonomy() -> dict:
    with open(TAG_TAXONOMY_JSON, 'r') as f:
        taxonomy = json.load(f)
    return {k: v for k, v in taxonomy.items() if not k.startswith("_")}


# ── FAISS Index (GPU only) ──────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS GPU index. Fatal if no GPU."""
    d = embeddings.shape[1]
    n = embeddings.shape[0]
    print(f"Building FAISS GPU index: {n:,} vectors, dim={d}")

    if faiss.get_num_gpus() == 0:
        print("FATAL: No FAISS GPU available. Install faiss-gpu.")
        sys.exit(1)

    res = faiss.StandardGpuResources()

    if n < 1000:
        index_cpu = faiss.IndexFlatL2(d)
    else:
        nlist = min(100, n // 10)
        quantizer = faiss.IndexFlatL2(d)
        index_cpu = faiss.IndexIVFFlat(quantizer, d, nlist)
        index_cpu.train(embeddings)

    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    index_gpu.add(embeddings)
    print(f"FAISS GPU index built: {index_gpu.ntotal:,} vectors")
    return index_gpu


# ── Clip Path Parsing (for Hard mode) ───────────────────────────────────

def parse_clip_metadata(clip_paths: list) -> list:
    """Extract (video_id, clip_index) from clip paths like .../video_id-NNN.mp4"""
    metadata = []
    for p in clip_paths:
        fname = Path(p).stem
        parts = fname.rsplit("-", 1)
        if len(parts) == 2:
            video_id = parts[0]
            try:
                clip_idx = int(parts[1])
            except ValueError:
                clip_idx = 0
        else:
            video_id = fname
            clip_idx = 0
        metadata.append((video_id, clip_idx))
    return metadata


def build_exclusion_mask(clip_metadata: list, window_sec: float = 30,
                         clip_duration_sec: float = 10) -> dict:
    """Build {query_idx: set(excluded_neighbor_indices)} for Hard mode."""
    window_clips = int(window_sec / clip_duration_sec)

    video_groups = {}
    for i, (vid, cidx) in enumerate(clip_metadata):
        video_groups.setdefault(vid, []).append((i, cidx))

    exclusion = {}
    for i, (vid, cidx) in enumerate(clip_metadata):
        excluded = set()
        for j, jcidx in video_groups[vid]:
            if j != i and abs(jcidx - cidx) <= window_clips:
                excluded.add(j)
        if excluded:
            exclusion[i] = excluded

    n_with = len(exclusion)
    avg_ex = np.mean([len(v) for v in exclusion.values()]) if exclusion else 0
    print(f"Hard mode exclusion: {n_with:,} clips affected (avg {avg_ex:.1f} excluded neighbors)")
    return exclusion


def apply_hard_filter(distances: np.ndarray, indices: np.ndarray,
                      exclusion: dict, k: int) -> tuple:
    """Filter kNN results by exclusion mask, keeping top-k valid neighbors."""
    n = indices.shape[0]
    hard_D = np.full((n, k + 1), np.inf, dtype=np.float32)
    hard_I = np.full((n, k + 1), -1, dtype=np.int64)

    for i in range(n):
        excluded = exclusion.get(i, set())
        kept = 0
        for j_pos in range(indices.shape[1]):
            neighbor = indices[i, j_pos]
            if neighbor == i:
                hard_D[i, 0] = distances[i, j_pos]
                hard_I[i, 0] = neighbor
                continue
            if neighbor not in excluded and kept < k:
                hard_D[i, kept + 1] = distances[i, j_pos]
                hard_I[i, kept + 1] = neighbor
                kept += 1
            if kept >= k:
                break

    return hard_D, hard_I


# ── Label-Free Metrics (3) ──────────────────────────────────────────────

def compute_cycle_at_k(indices: np.ndarray, k: int) -> tuple:
    """Cycle@K: % of clips where kNN(A)=B implies B's kNN includes A.
    Returns (global_score, per_clip_bool_array) for groupby analysis."""
    n = indices.shape[0]
    per_clip = np.full(n, -1, dtype=np.int8)  # -1=invalid, 0=fail, 1=success

    for i in range(n):
        nearest = indices[i, 1]
        if nearest < 0 or nearest >= n:
            continue
        per_clip[i] = 1 if i in set(indices[nearest, 1:k + 1].tolist()) else 0

    valid_mask = per_clip >= 0
    global_score = (per_clip[valid_mask].sum() / valid_mask.sum() * 100) if valid_mask.sum() > 0 else 0.0
    return float(global_score), per_clip


def compute_overlap_at_k(embeddings: np.ndarray, k: int) -> float:
    """
    Overlap@K: IoU of neighborhoods from two embedding "views".
    Approximation: split embedding dims into two halves, compare kNN sets.
    (True Overlap@K requires augmented video crops — not available in this pipeline.)
    """
    n, d = embeddings.shape
    if n < 100:
        return None

    mid = d // 2
    emb_a = np.ascontiguousarray(embeddings[:, :mid].astype(np.float32))
    emb_b = np.ascontiguousarray(embeddings[:, mid:].astype(np.float32))

    # Sample for large datasets (O(n²) search)
    rng = np.random.RandomState(42)
    sample_n = min(n, 5000)
    sample_idx = rng.choice(n, sample_n, replace=False) if n > sample_n else np.arange(n)

    res = faiss.StandardGpuResources()
    idx_a_cpu = faiss.IndexFlatL2(mid)
    idx_b_cpu = faiss.IndexFlatL2(d - mid)
    idx_a = faiss.index_cpu_to_gpu(res, 0, idx_a_cpu)
    idx_b = faiss.index_cpu_to_gpu(res, 0, idx_b_cpu)
    idx_a.add(emb_a)
    idx_b.add(emb_b)

    _, I_a = idx_a.search(emb_a[sample_idx], k + 1)
    _, I_b = idx_b.search(emb_b[sample_idx], k + 1)

    per_clip_iou = np.zeros(sample_n)
    for row in range(sample_n):
        set_a = set(I_a[row, 1:k + 1].tolist()) - {-1}
        set_b = set(I_b[row, 1:k + 1].tolist()) - {-1}
        union = set_a | set_b
        if union:
            per_clip_iou[row] = len(set_a & set_b) / len(union)

    from utils.bootstrap import bootstrap_ci
    ci = bootstrap_ci(per_clip_iou * 100)
    return float(np.mean(per_clip_iou)) * 100, ci


def compute_silhouette(embeddings: np.ndarray, tags: list,
                       field: str = "scene_type") -> float:
    """Silhouette score on embeddings grouped by `field`. Returns [-1, 1]."""
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("WARNING: sklearn not available, skipping silhouette")
        return 0.0

    labels = [t.get(field, "unknown") for t in tags]
    unique = set(labels)
    if len(unique) < 2:
        return 0.0

    n = len(embeddings)
    # silhouette_score requires 2 <= n_labels <= n_samples - 1
    if len(unique) >= n:
        print(f"  Silhouette: SKIPPED (n_labels={len(unique)} >= n_samples={n})")
        return 0.0

    label_map = {l: i for i, l in enumerate(sorted(unique))}
    label_ints = np.array([label_map[l] for l in labels])

    if n > 10000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, 10000, replace=False)
        return float(silhouette_score(embeddings[idx], label_ints[idx], metric="cosine"))

    return float(silhouette_score(embeddings, label_ints, metric="cosine"))


# ── Pseudo-Label Metrics (3) ────────────────────────────────────────────

def compute_prec_at_k(indices: np.ndarray, tags: list, k: int,
                      field: str = "scene_type") -> float:
    """Prec@K: % of kNN neighbors with same tag value."""
    correct = 0
    total = 0
    n_tags = len(tags)

    for i, neighbors in enumerate(indices):
        my_val = tags[i].get(field, "unknown")
        for j in neighbors[1:k + 1]:
            if 0 <= j < n_tags:
                if tags[j].get(field, "unknown") == my_val:
                    correct += 1
                total += 1

    return (correct / total * 100) if total > 0 else 0.0


def compute_map_at_k(indices: np.ndarray, tags: list, k: int,
                     field: str = "scene_type") -> float:
    """mAP@K: Mean Average Precision using `field` as relevance."""
    n = indices.shape[0]
    n_tags = len(tags)
    total_ap = 0.0

    for i in range(n):
        my_val = tags[i].get(field, "unknown")
        hits = 0
        ap_sum = 0.0

        for rank in range(1, k + 1):
            j = indices[i, rank]
            if j < 0 or j >= n_tags:
                continue
            if tags[j].get(field, "unknown") == my_val:
                hits += 1
                ap_sum += hits / rank

        total_ap += ap_sum / k

    return total_ap / n if n > 0 else 0.0


def compute_ndcg_at_k(indices: np.ndarray, tags: list, k: int,
                      taxonomy: dict) -> float:
    """nDCG@K: graded relevance from multi-field tag overlap (single-value fields)."""
    single_fields = [f for f, spec in taxonomy.items() if spec["type"] == "single"]
    n_fields = len(single_fields)
    n = indices.shape[0]
    n_tags = len(tags)
    total_ndcg = 0.0
    valid = 0

    for i in range(n):
        grades = []
        for rank in range(1, k + 1):
            j = indices[i, rank]
            if j < 0 or j >= n_tags:
                grades.append(0.0)
                continue
            matching = sum(1 for f in single_fields
                          if tags[i].get(f) is not None and tags[i].get(f) == tags[j].get(f))
            grades.append(matching / n_fields)

        dcg = sum(g / np.log2(r + 2) for r, g in enumerate(grades))
        ideal = sorted(grades, reverse=True)
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))

        if idcg > 0:
            total_ndcg += dcg / idcg
            valid += 1

    return total_ndcg / valid if valid > 0 else 0.0


# ── Analysis Metrics (3) ────────────────────────────────────────────────

def compute_per_scene_purity(indices: np.ndarray, tags: list, k: int) -> dict:
    """Prec@K breakdown per scene_type."""
    stats = {}
    n_tags = len(tags)

    for i, neighbors in enumerate(indices):
        my_type = tags[i].get("scene_type", "unknown")
        stats.setdefault(my_type, {"correct": 0, "total": 0, "count": 0})
        stats[my_type]["count"] += 1

        for j in neighbors[1:k + 1]:
            if 0 <= j < n_tags:
                if tags[j].get("scene_type", "unknown") == my_type:
                    stats[my_type]["correct"] += 1
                stats[my_type]["total"] += 1

    return {scene: {"prec_at_k": round(s["correct"] / s["total"] * 100, 2), "count": s["count"]}
            for scene, s in sorted(stats.items()) if s["total"] > 0}


def compute_multi_attribute_slices(indices: np.ndarray, tags: list, k: int) -> dict:
    """Prec@K (scene_type match) grouped by each slice field's values."""
    slices = {}
    n_tags = len(tags)

    for field in SLICE_FIELDS:
        field_stats = {}
        for i, neighbors in enumerate(indices):
            my_val = tags[i].get(field, "unknown")
            field_stats.setdefault(my_val, {"correct": 0, "total": 0, "count": 0})
            field_stats[my_val]["count"] += 1

            my_scene = tags[i].get("scene_type", "unknown")
            for j in neighbors[1:k + 1]:
                if 0 <= j < n_tags:
                    if tags[j].get("scene_type", "unknown") == my_scene:
                        field_stats[my_val]["correct"] += 1
                    field_stats[my_val]["total"] += 1

        slices[field] = {val: {"prec_at_k": round(s["correct"] / s["total"] * 100, 2), "count": s["count"]}
                         for val, s in sorted(field_stats.items()) if s["total"] > 0}
    return slices


def compute_confidence_sweep(indices: np.ndarray, tags: list, k: int) -> list:
    """Vary confidence threshold → Prec@K vs coverage."""
    n = len(tags)
    n_tags = n
    results = []

    for thresh in CONFIDENCE_THRESHOLDS:
        mask = [i for i, t in enumerate(tags)
                if isinstance(t.get("confidence_scene_type"), (int, float))
                and t["confidence_scene_type"] >= thresh]

        coverage = len(mask) / n if n > 0 else 0.0

        if len(mask) < 10:
            results.append({"threshold": thresh, "coverage": round(coverage, 4),
                            "prec_at_k": None, "n_clips": len(mask)})
            continue

        correct = total = 0
        for i in mask:
            my_type = tags[i].get("scene_type", "unknown")
            for j in indices[i, 1:k + 1]:
                if 0 <= j < n_tags:
                    if tags[j].get("scene_type", "unknown") == my_type:
                        correct += 1
                    total += 1

        results.append({"threshold": thresh, "coverage": round(coverage, 4),
                        "prec_at_k": round(correct / total * 100, 2) if total > 0 else 0.0,
                        "n_clips": len(mask)})
    return results


def compute_macro_micro_avg(per_scene: dict) -> tuple:
    """Macro (equal per-class weight) and micro (count-weighted) averaging."""
    if not per_scene:
        return {"prec_at_k": 0.0}, {"prec_at_k": 0.0}

    macro = {"prec_at_k": round(sum(s["prec_at_k"] for s in per_scene.values()) / len(per_scene), 2)}

    total_w = sum(s["prec_at_k"] * s["count"] for s in per_scene.values())
    total_c = sum(s["count"] for s in per_scene.values())
    micro = {"prec_at_k": round(total_w / total_c, 2) if total_c > 0 else 0.0}

    return macro, micro


def compute_cycle_per_key(cycle_per_clip: np.ndarray, tags: list,
                          field: str) -> dict:
    """Group per-clip cycle results by a tag field → {value: {cycle_at_k, count}}."""
    groups = {}
    for i, t in enumerate(tags):
        val = t.get(field, "unknown")
        groups.setdefault(val, [])
        if cycle_per_clip[i] >= 0:
            groups[val].append(cycle_per_clip[i])

    return {val: {"cycle_at_k": round(int(sum(int(x) for x in arr)) / len(arr) * 100, 2),
                  "count": len(arr)}
            for val, arr in sorted(groups.items()) if len(arr) > 0}


def compute_map_per_key(indices: np.ndarray, tags: list, k: int,
                        field: str) -> dict:
    """mAP@K grouped by each value of `field`."""
    n_tags = len(tags)
    groups = {}
    for i in range(indices.shape[0]):
        val = tags[i].get(field, "unknown")
        groups.setdefault(val, {"ap_sum": 0.0, "count": 0})
        groups[val]["count"] += 1

        my_val = tags[i].get(field, "unknown")
        hits = 0
        ap = 0.0
        for rank in range(1, k + 1):
            j = indices[i, rank]
            if j < 0 or j >= n_tags:
                continue
            if tags[j].get(field, "unknown") == my_val:
                hits += 1
                ap += hits / rank
        groups[val]["ap_sum"] += ap / k

    return {val: {"map_at_k": round(s["ap_sum"] / s["count"], 4), "count": s["count"]}
            for val, s in sorted(groups.items()) if s["count"] > 0}


# ── Compute All Metrics for a Mode ──────────────────────────────────────

def compute_all_metrics(embeddings: np.ndarray, D: np.ndarray, I: np.ndarray,
                        tags: list, k: int, taxonomy: dict, mode: str) -> dict:
    """Compute all 9 metrics for one mode (easy or hard)."""
    single_fields = [f for f, spec in taxonomy.items() if spec["type"] == "single"]
    print(f"\n{'='*50}")
    print(f"{mode.upper()} MODE (k={k})")
    print(f"{'='*50}")

    t0 = time.time()

    # ── Global metrics ──
    cycle, cycle_per_clip = compute_cycle_at_k(I, k)
    print(f"  Cycle@K:     {cycle:.2f}%")

    overlap_result = compute_overlap_at_k(embeddings, k)
    if overlap_result is not None:
        overlap, overlap_ci = overlap_result
        print(f"  Overlap@K:   {overlap:.2f}% \u00b1 {overlap_ci['ci_half']:.2f} (dim-split approx)")
    else:
        overlap, overlap_ci = None, None
        print(f"  Overlap@K:   SKIPPED (< 100 clips)")

    sil = compute_silhouette(embeddings, tags)
    print(f"  Silhouette:  {sil:.4f} (scene_type)")

    prec = compute_prec_at_k(I, tags, k)
    print(f"  Prec@K:      {prec:.2f}%")

    map_k = compute_map_at_k(I, tags, k)
    print(f"  mAP@K:       {map_k:.4f}")

    ndcg = compute_ndcg_at_k(I, tags, k, taxonomy)
    print(f"  nDCG@K:      {ndcg:.4f}")

    # ── Per-key breakdowns (Step 11: repeat for all single-value fields) ──
    print(f"  Computing per-key breakdowns for {len(single_fields)} fields...")

    # Silhouette per-key: which dimension does V-JEPA cluster by?
    silhouette_per_key = {}
    for field in single_fields:
        s = compute_silhouette(embeddings, tags, field=field)
        silhouette_per_key[field] = round(s, 4)
    sil_sorted = sorted(silhouette_per_key.items(), key=lambda x: x[1], reverse=True)
    print(f"  Silhouette per-key: {', '.join(f'{f}={v:.4f}' for f, v in sil_sorted[:3])} ...")

    # Prec@K per-key: retrieval purity using each field as match criterion
    prec_per_key = {}
    for field in single_fields:
        prec_per_key[field] = round(compute_prec_at_k(I, tags, k, field=field), 2)
    prec_sorted = sorted(prec_per_key.items(), key=lambda x: x[1], reverse=True)
    print(f"  Prec@K per-key: {', '.join(f'{f}={v:.2f}%' for f, v in prec_sorted[:3])} ...")

    # mAP@K per-key: ranking quality using each field as relevance
    map_per_key = {}
    for field in single_fields:
        map_per_key[field] = round(compute_map_at_k(I, tags, k, field=field), 4)
    map_sorted = sorted(map_per_key.items(), key=lambda x: x[1], reverse=True)
    print(f"  mAP@K per-key: {', '.join(f'{f}={v:.4f}' for f, v in map_sorted[:3])} ...")

    # Cycle@K per-key: neighborhood coherence grouped by tag values
    cycle_per_key = {}
    for field in single_fields:
        cycle_per_key[field] = compute_cycle_per_key(cycle_per_clip, tags, field)
    print(f"  Cycle@K per-key: computed for {len(cycle_per_key)} fields")

    # mAP@K per-value: mAP@K broken down by each value within each field
    map_per_value = {}
    for field in single_fields:
        map_per_value[field] = compute_map_per_key(I, tags, k, field)

    per_scene = compute_per_scene_purity(I, tags, k)
    multi_attr = compute_multi_attribute_slices(I, tags, k)
    macro, micro = compute_macro_micro_avg(per_scene)
    print(f"  Macro avg:   {macro['prec_at_k']:.2f}%")
    print(f"  Micro avg:   {micro['prec_at_k']:.2f}%")

    # ── Bootstrap 95% CI on global metrics ──
    print(f"  Bootstrap 95% CI (BCa, 10K iterations)...")
    pc_prec = per_clip_prec_at_k(I, tags, k)
    pc_map = per_clip_map_at_k(I, tags, k)
    pc_cycle = per_clip_cycle_at_k(I, k)
    pc_ndcg = per_clip_ndcg_at_k(I, tags, k, taxonomy)

    ci_prec = bootstrap_ci(pc_prec)
    ci_map = bootstrap_ci(pc_map)
    ci_cycle = bootstrap_ci(pc_cycle[~np.isnan(pc_cycle)])
    ci_ndcg = bootstrap_ci(pc_ndcg[~np.isnan(pc_ndcg)])

    print(f"  Prec@K:  {ci_prec['mean']:.2f}% ± {ci_prec['ci_half']:.2f}")
    print(f"  mAP@K:   {ci_map['mean']:.4f} ± {ci_map['ci_half']:.4f}")
    print(f"  Cycle@K: {ci_cycle['mean']:.2f}% ± {ci_cycle['ci_half']:.2f}")
    print(f"  nDCG@K:  {ci_ndcg['mean']:.4f} ± {ci_ndcg['ci_half']:.4f}")
    print(f"  [{time.time() - t0:.1f}s]")

    return {
        "cycle_at_k": round(cycle, 2),
        "overlap_at_k": round(overlap, 2) if overlap is not None else None,
        "silhouette": round(sil, 4),
        "prec_at_k": round(prec, 2),
        "map_at_k": round(map_k, 4),
        "ndcg_at_k": round(ndcg, 4),
        "ci": {
            "prec_at_k": ci_prec,
            "map_at_k": ci_map,
            "cycle_at_k": ci_cycle,
            "ndcg_at_k": ci_ndcg,
            **({"overlap_at_k": overlap_ci} if overlap_ci else {}),
        },
        "per_scene": per_scene,
        "multi_attribute_slices": multi_attr,
        "silhouette_per_key": silhouette_per_key,
        "prec_per_key": prec_per_key,
        "map_per_key": map_per_key,
        "cycle_per_key": cycle_per_key,
        "map_per_value": map_per_value,
        "macro_avg": macro,
        "micro_avg": micro,
    }


# ── Plots ────────────────────────────────────────────────────────────────

def generate_plots(easy: dict, hard: dict, conf_sweep: list,
                   D_easy: np.ndarray, k: int, output_dir: Path, n: int,
                   sfx: str = ""):
    """Generate diagnostic plots (.png + .pdf)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: kNN distance distribution (clipped to 99.9th pctile) ─
    fig, ax = plt.subplots(figsize=(10, 6))
    dists = D_easy[:, 1:].flatten()
    dists = dists[np.isfinite(dists) & (dists > 0)]
    clip_val = np.percentile(dists, 99.9)
    dists_clipped = dists[dists <= clip_val]
    ax.hist(dists_clipped, bins=80, color='steelblue', alpha=0.7, edgecolor='white')
    med = np.median(dists_clipped)
    ax.axvline(x=med, color='red', linestyle='--', label=f'Median: {med:.2f}')
    ax.set_xlabel('L2 Distance')
    ax.set_ylabel('Count')
    ax.set_title(f'kNN Distance Distribution (k={k}, clipped to 99.9th pctile)')
    ax.legend()
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m06_distance_hist{sfx}{ext}", dpi=150 if ext == ".png" else None)
    plt.close()
    print(f"Saved: {output_dir / 'm06_distance_hist.png'}")

    # ── Plot 3: Confidence sweep ─────────────────────────────────────
    valid_sw = [s for s in conf_sweep if s["prec_at_k"] is not None]
    if valid_sw:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ts = [s["threshold"] for s in valid_sw]
        pv = [s["prec_at_k"] for s in valid_sw]
        cv = [s["coverage"] * 100 for s in valid_sw]

        c1, c2 = '#1f77b4', '#ff7f0e'
        ax1.plot(ts, pv, 'o-', color=c1, linewidth=2, label='Prec@K')
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Prec@K (%)', color=c1)
        ax1.tick_params(axis='y', labelcolor=c1)

        ax2 = ax1.twinx()
        ax2.plot(ts, cv, 's--', color=c2, linewidth=2, label='Coverage')
        ax2.set_ylabel('Coverage (%)', color=c2)
        ax2.tick_params(axis='y', labelcolor=c2)

        ax1.set_title(f'Confidence Sweep: Prec@K vs Coverage (k={k})')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='center left')
        plt.tight_layout()
        for ext in [".png", ".pdf"]:
            plt.savefig(output_dir / f"m06_confidence_sweep{sfx}{ext}", dpi=150 if ext == ".png" else None)
        plt.close()
        print(f"Saved: {output_dir / 'm06_confidence_sweep.png'}")

    # ── Plot 4: Multi-panel purity (all 9 single-value taxonomy keys) ─
    panel_data = [("scene_type", easy["per_scene"], hard["per_scene"])]
    for field in SLICE_FIELDS:
        e_slice = easy.get("multi_attribute_slices", {}).get(field, {})
        h_slice = hard.get("multi_attribute_slices", {}).get(field, {})
        if e_slice:
            panel_data.append((field, e_slice, h_slice))

    n_panels = len(panel_data)
    n_cols_grid = 3
    n_rows_grid = (n_panels + n_cols_grid - 1) // n_cols_grid
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(6 * n_cols_grid, 5 * n_rows_grid))
    axes_flat = axes.flatten() if n_panels > 1 else [axes]

    for idx, (field_name, e_data, h_data) in enumerate(panel_data):
        ax = axes_flat[idx]
        all_vals = sorted(set(list(e_data.keys()) + list(h_data.keys())))
        xp = np.arange(len(all_vals))
        wp = 0.35
        ev_p = [e_data.get(v, {}).get("prec_at_k", 0) for v in all_vals]
        hv_p = [h_data.get(v, {}).get("prec_at_k", 0) for v in all_vals]
        counts = [e_data.get(v, {}).get("count", 0) for v in all_vals]

        bars = ax.bar(xp - wp / 2, ev_p, wp, color="#4CAF50", alpha=0.8)
        ax.bar(xp + wp / 2, hv_p, wp, color="#F44336", alpha=0.8)

        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'n={cnt}', ha='center', va='bottom', fontsize=6)

        label_name = field_name.replace("_", " ").title()
        ax.set_title(label_name, fontsize=11, fontweight='bold')
        ax.set_xticks(xp)
        ax.set_xticklabels(all_vals, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, max(max(ev_p, default=0), max(hv_p, default=0)) * 1.3 + 5)
        ax.set_ylabel('Prec@K (%)', fontsize=8)

    # Hide unused axes
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4CAF50', alpha=0.8, label='Easy'),
                       Patch(facecolor='#F44336', alpha=0.8, label='Hard')]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    fig.suptitle(f'Retrieval Purity — All Taxonomy Keys (k={k}, n={n:,})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m06_purity_all_keys{sfx}{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'm06_purity_all_keys.png'}")

    # ── Plot 5: Silhouette per-key bar chart ─────────────────────────
    e_sil = easy.get("silhouette_per_key", {})
    h_sil = hard.get("silhouette_per_key", {})
    if e_sil:
        fields_sorted = sorted(e_sil.keys(), key=lambda f: e_sil.get(f, 0), reverse=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        xp = np.arange(len(fields_sorted))
        wp = 0.35
        ev = [e_sil.get(f, 0) for f in fields_sorted]
        hv = [h_sil.get(f, 0) for f in fields_sorted]
        ax.bar(xp - wp / 2, ev, wp, color="#4CAF50", alpha=0.8, label="Easy")
        ax.bar(xp + wp / 2, hv, wp, color="#F44336", alpha=0.8, label="Hard")
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xticks(xp)
        ax.set_xticklabels([f.replace("_", " ").title() for f in fields_sorted],
                           rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Silhouette Score')
        ax.set_title(f'Silhouette per Taxonomy Key — Which Dimension Does V-JEPA Cluster By? (k={k}, n={n:,})',
                     fontsize=11, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        for ext in [".png", ".pdf"]:
            plt.savefig(output_dir / f"m06_silhouette_per_key{sfx}{ext}",
                        dpi=150 if ext == ".png" else None, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'm06_silhouette_per_key.png'}")

    # ── Plot 6: mAP@K per-key bar chart ──────────────────────────────
    e_map = easy.get("map_per_key", {})
    h_map = hard.get("map_per_key", {})
    if e_map:
        fields_sorted = sorted(e_map.keys(), key=lambda f: e_map.get(f, 0), reverse=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        xp = np.arange(len(fields_sorted))
        wp = 0.35
        ev = [e_map.get(f, 0) for f in fields_sorted]
        hv = [h_map.get(f, 0) for f in fields_sorted]
        ax.bar(xp - wp / 2, ev, wp, color="#4CAF50", alpha=0.8, label="Easy")
        ax.bar(xp + wp / 2, hv, wp, color="#F44336", alpha=0.8, label="Hard")
        ax.set_xticks(xp)
        ax.set_xticklabels([f.replace("_", " ").title() for f in fields_sorted],
                           rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('mAP@K')
        ax.set_title(f'mAP@K per Taxonomy Key — Ranking Quality by Field (k={k}, n={n:,})',
                     fontsize=11, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        for ext in [".png", ".pdf"]:
            plt.savefig(output_dir / f"m06_map_per_key{sfx}{ext}",
                        dpi=150 if ext == ".png" else None, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'm06_map_per_key.png'}")

    # ── Plot 7: Cycle@K per-key 3x3 multi-panel ─────────────────────
    e_cycle = easy.get("cycle_per_key", {})
    h_cycle = hard.get("cycle_per_key", {})
    if e_cycle:
        cycle_fields = sorted(e_cycle.keys())
        n_cp = len(cycle_fields)
        n_cols_c = 3
        n_rows_c = (n_cp + n_cols_c - 1) // n_cols_c
        fig, axes = plt.subplots(n_rows_c, n_cols_c, figsize=(6 * n_cols_c, 5 * n_rows_c))
        axes_flat = axes.flatten() if n_cp > 1 else [axes]

        for idx, field_name in enumerate(cycle_fields):
            ax = axes_flat[idx]
            e_fd = e_cycle.get(field_name, {})
            h_fd = h_cycle.get(field_name, {})
            all_vals = sorted(set(list(e_fd.keys()) + list(h_fd.keys())))
            xp = np.arange(len(all_vals))
            wp = 0.35
            ev_c = [e_fd.get(v, {}).get("cycle_at_k", 0) for v in all_vals]
            hv_c = [h_fd.get(v, {}).get("cycle_at_k", 0) for v in all_vals]
            counts = [e_fd.get(v, {}).get("count", 0) for v in all_vals]

            bars = ax.bar(xp - wp / 2, ev_c, wp, color="#4CAF50", alpha=0.8)
            ax.bar(xp + wp / 2, hv_c, wp, color="#F44336", alpha=0.8)
            for bar, cnt in zip(bars, counts):
                if cnt > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'n={cnt}', ha='center', va='bottom', fontsize=6)
            ax.set_title(field_name.replace("_", " ").title(), fontsize=11, fontweight='bold')
            ax.set_xticks(xp)
            ax.set_xticklabels(all_vals, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, max(max(ev_c, default=0), max(hv_c, default=0)) * 1.3 + 5)
            ax.set_ylabel('Cycle@K (%)', fontsize=8)

        for idx in range(n_cp, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        from matplotlib.patches import Patch as _Patch
        fig.legend(handles=[_Patch(facecolor='#4CAF50', alpha=0.8, label='Easy'),
                            _Patch(facecolor='#F44336', alpha=0.8, label='Hard')],
                   loc='upper right', fontsize=10, framealpha=0.9)
        fig.suptitle(f'Cycle@K per Taxonomy Key (k={k}, n={n:,})', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        for ext in [".png", ".pdf"]:
            plt.savefig(output_dir / f"m06_cycle_per_key{sfx}{ext}",
                        dpi=150 if ext == ".png" else None, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'm06_cycle_per_key.png'}")

    # ── Plot 8: mAP@K per-value 3x3 multi-panel ─────────────────────
    e_mapv = easy.get("map_per_value", {})
    h_mapv = hard.get("map_per_value", {})
    if e_mapv:
        mapv_fields = sorted(e_mapv.keys())
        n_mp = len(mapv_fields)
        n_cols_m = 3
        n_rows_m = (n_mp + n_cols_m - 1) // n_cols_m
        fig, axes = plt.subplots(n_rows_m, n_cols_m, figsize=(6 * n_cols_m, 5 * n_rows_m))
        axes_flat = axes.flatten() if n_mp > 1 else [axes]

        for idx, field_name in enumerate(mapv_fields):
            ax = axes_flat[idx]
            e_fd = e_mapv.get(field_name, {})
            h_fd = h_mapv.get(field_name, {})
            all_vals = sorted(set(list(e_fd.keys()) + list(h_fd.keys())))
            xp = np.arange(len(all_vals))
            wp = 0.35
            ev_m = [e_fd.get(v, {}).get("map_at_k", 0) for v in all_vals]
            hv_m = [h_fd.get(v, {}).get("map_at_k", 0) for v in all_vals]
            counts = [e_fd.get(v, {}).get("count", 0) for v in all_vals]

            bars = ax.bar(xp - wp / 2, ev_m, wp, color="#4CAF50", alpha=0.8)
            ax.bar(xp + wp / 2, hv_m, wp, color="#F44336", alpha=0.8)
            for bar, cnt in zip(bars, counts):
                if cnt > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                            f'n={cnt}', ha='center', va='bottom', fontsize=6)
            ax.set_title(field_name.replace("_", " ").title(), fontsize=11, fontweight='bold')
            ax.set_xticks(xp)
            ax.set_xticklabels(all_vals, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, max(max(ev_m, default=0), max(hv_m, default=0)) * 1.3 + 0.05)
            ax.set_ylabel('mAP@K', fontsize=8)

        for idx in range(n_mp, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        from matplotlib.patches import Patch as _Patch2
        fig.legend(handles=[_Patch2(facecolor='#4CAF50', alpha=0.8, label='Easy'),
                            _Patch2(facecolor='#F44336', alpha=0.8, label='Hard')],
                   loc='upper right', fontsize=10, framealpha=0.9)
        fig.suptitle(f'mAP@K per Taxonomy Value (k={k}, n={n:,})', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        for ext in [".png", ".pdf"]:
            plt.savefig(output_dir / f"m06_map_per_value{sfx}{ext}",
                        dpi=150 if ext == ".png" else None, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'm06_map_per_value.png'}")

    # ── Plot 9: Combined metrics-by-key grid (1x3) ──────────────────
    e_sil2 = easy.get("silhouette_per_key", {})
    e_prec2 = easy.get("prec_per_key", {})
    e_map2 = easy.get("map_per_key", {})
    h_sil2 = hard.get("silhouette_per_key", {})
    h_prec2 = hard.get("prec_per_key", {})
    h_map2 = hard.get("map_per_key", {})
    if e_sil2 and e_prec2 and e_map2:
        # Sort all panels by mAP (strongest signal) for consistent x-axis
        fields_sorted = sorted(e_map2.keys(), key=lambda f: e_map2.get(f, 0), reverse=True)
        xlabels = [f.replace("_", " ").title() for f in fields_sorted]
        xp = np.arange(len(fields_sorted))
        wp = 0.35

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        # Panel 1: Silhouette
        ax = axes[0]
        ev = [e_sil2.get(f, 0) for f in fields_sorted]
        hv = [h_sil2.get(f, 0) for f in fields_sorted]
        ax.bar(xp - wp / 2, ev, wp, color="#4CAF50", alpha=0.8, label="Easy")
        ax.bar(xp + wp / 2, hv, wp, color="#F44336", alpha=0.8, label="Hard")
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_xticks(xp)
        ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette per Key\n(higher = better clustering, range [-1, +1])',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)

        # Panel 2: Prec@K
        ax = axes[1]
        ev = [e_prec2.get(f, 0) for f in fields_sorted]
        hv = [h_prec2.get(f, 0) for f in fields_sorted]
        ax.bar(xp - wp / 2, ev, wp, color="#4CAF50", alpha=0.8, label="Easy")
        ax.bar(xp + wp / 2, hv, wp, color="#F44336", alpha=0.8, label="Hard")
        ax.set_xticks(xp)
        ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Prec@K (%)')
        ax.set_title('Prec@K per Key\n(higher = more neighbors match, %)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)

        # Panel 3: mAP@K
        ax = axes[2]
        ev = [e_map2.get(f, 0) for f in fields_sorted]
        hv = [h_map2.get(f, 0) for f in fields_sorted]
        ax.bar(xp - wp / 2, ev, wp, color="#4CAF50", alpha=0.8, label="Easy")
        ax.bar(xp + wp / 2, hv, wp, color="#F44336", alpha=0.8, label="Hard")
        ax.set_xticks(xp)
        ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('mAP@K')
        ax.set_title('mAP@K per Key\n(higher = better ranked results, range [0, 1])',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)

        fig.suptitle(f'V-JEPA Representation Quality — All Metrics by Taxonomy Key (k={k}, n={n:,})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        for ext in [".png", ".pdf"]:
            plt.savefig(output_dir / f"m06_metrics_by_key{sfx}{ext}",
                        dpi=150 if ext == ".png" else None, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'm06_metrics_by_key.png'}")

    # ── Plot 10: Radar — Easy vs Hard ────────────────────────────────
    radar_keys = ["cycle_at_k", "prec_at_k"]
    if easy.get("overlap_at_k") is not None:
        radar_keys.append("overlap_at_k")
    # silhouette is [-1,1], scale to [0,100] for radar only
    radar_labels, re, rh = [], [], []
    for m in radar_keys:
        if easy.get(m) is not None and hard.get(m) is not None:
            radar_labels.append(m.replace("_", " ").title())
            re.append(easy[m])
            rh.append(hard[m])
    # Add silhouette scaled
    radar_labels.append("Silhouette")
    re.append((easy["silhouette"] + 1) / 2 * 100)
    rh.append((hard["silhouette"] + 1) / 2 * 100)
    # mAP and nDCG scaled to percentage
    for m in ["map_at_k", "ndcg_at_k"]:
        radar_labels.append(m.replace("_", " ").title())
        re.append(easy[m] * 100)
        rh.append(hard[m] * 100)

    if len(radar_labels) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        re_c = re + [re[0]]
        rh_c = rh + [rh[0]]
        angles_c = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles_c, re_c, 'o-', linewidth=2, label='Easy', color='#4CAF50')
        ax.fill(angles_c, re_c, alpha=0.1, color='#4CAF50')
        ax.plot(angles_c, rh_c, 's-', linewidth=2, label='Hard', color='#F44336')
        ax.fill(angles_c, rh_c, alpha=0.1, color='#F44336')
        ax.set_xticks(angles)
        ax.set_xticklabels(radar_labels, fontsize=9)
        ax.set_title(f'Easy vs Hard (k={k})', pad=20)
        ax.legend(loc='upper right')
        plt.tight_layout()
        for ext in [".png", ".pdf"]:
            plt.savefig(output_dir / f"m06_radar{sfx}{ext}", dpi=150 if ext == ".png" else None)
        plt.close()
        print(f"Saved: {output_dir / 'm06_radar.png'}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute 9 evaluation metrics via FAISS-GPU (Easy/Hard)")
    parser.add_argument("--SANITY", action="store_true", help="Run on first 100 clips")
    parser.add_argument("--FULL", action="store_true", help="Run on all clips")
    parser.add_argument("--k", type=int, default=FAISS_K_NEIGHBORS,
                        help=f"Number of neighbors (default: {FAISS_K_NEIGHBORS})")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--true-overlap", action="store_true",
                        help="Use augmented embeddings for True Overlap@K (from m05c)")
    add_encoder_arg(parser)
    add_subset_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    check_gpu()

    mode = "SANITY" if args.SANITY else ("POC" if args.subset else "FULL")
    wb_run = init_wandb("m06", mode, config=vars(args),
                        enabled=not args.no_wandb)

    # Output routing: POC vs Full, encoder-aware paths
    output_dir = get_output_dir(args.subset, sanity=args.SANITY)
    enc_files = get_encoder_files(args.encoder, output_dir)
    metrics_file = enc_files["metrics"]
    emb_file = enc_files["embeddings"]
    paths_file = enc_files["paths"]

    # Tags are shared across all encoders (not encoder-specific)
    tags_file = output_dir / "tags.json"

    enc_info = get_encoder_info(args.encoder)
    print(f"Encoder:     {args.encoder} (dim={enc_info['dim']}, type={enc_info['type']})")
    print(f"Output dir:  {output_dir}")
    print(f"Embeddings:  {emb_file}")
    print(f"Metrics:     {metrics_file}")
    if args.subset:
        print(f"[POC] Subset: {args.subset}")

    for f, desc in [(emb_file, "embeddings"), (tags_file, "tags")]:
        if not f.exists():
            print(f"FATAL: {desc} not found: {f}")
            sys.exit(1)

    embeddings = np.load(emb_file).astype(np.float32)
    with open(tags_file) as f:
        tags = json.load(f)
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded tags: {len(tags):,}")

    # Clip paths for Hard mode
    clip_paths = []
    if paths_file.exists():
        clip_paths = np.load(paths_file, allow_pickle=True).tolist()
        print(f"Loaded clip paths: {len(clip_paths):,}")

    # Align + key validation
    n = min(embeddings.shape[0], len(tags))
    if len(tags) != embeddings.shape[0]:
        print(f"WARNING: Mismatch {embeddings.shape[0]} emb vs {len(tags)} tags, truncating to {n}")
    embeddings = embeddings[:n]
    tags = tags[:n]
    if clip_paths:
        clip_paths = clip_paths[:n]
        # Verify embedding-tag alignment via clip keys
        tag_keys = [t.get("_clip_key", "") for t in tags]
        if tag_keys[0]:  # tags have _clip_key field
            mismatches = sum(1 for p, k in zip(clip_paths, tag_keys) if p != k)
            if mismatches > 0:
                print(f"FATAL: {mismatches}/{n} key mismatches between embeddings.paths.npy and tags.json")
                print(f"  First emb key: {clip_paths[0]}")
                print(f"  First tag key: {tag_keys[0]}")
                sys.exit(1)
            print(f"Key alignment verified: {n:,} clips match")

    if args.SANITY:
        n = min(100, n)
        embeddings = embeddings[:n]
        tags = tags[:n]
        clip_paths = clip_paths[:n] if clip_paths else []
        print(f"SANITY MODE: {n} clips")

    k = min(args.k, n - 1)
    taxonomy = load_taxonomy()

    # FAISS GPU index
    index = build_faiss_index(embeddings)

    # Search with extra neighbors for Hard mode filtering
    k_search = min(k * 5, n - 1)
    print(f"\nSearching {k_search} neighbors (k={k}, extra for Hard mode)...")
    D, I = index.search(embeddings, k_search + 1)

    # ── Easy mode ────────────────────────────────────────────────────
    easy = compute_all_metrics(embeddings, D[:, :k + 1], I[:, :k + 1], tags, k, taxonomy, "easy")

    # ── Hard mode ────────────────────────────────────────────────────
    if clip_paths:
        meta = parse_clip_metadata(clip_paths)
        exclusion = build_exclusion_mask(meta, EXCLUSION_WINDOW_SEC, DEFAULT_CLIP_DURATION_SEC)
        D_hard, I_hard = apply_hard_filter(D, I, exclusion, k)
        hard = compute_all_metrics(embeddings, D_hard, I_hard, tags, k, taxonomy, "hard")
    else:
        print("\nWARNING: No clip paths → Hard mode = Easy mode (no exclusion)")
        hard = easy

    # ── True Overlap@K (replaces dim-split if augmented embeddings exist) ──
    if args.true_overlap:
        result = compute_true_overlap_at_k(output_dir, k)
        if result is not None:
            true_ov, overlap_ci = result
            print(f"\n  True Overlap@K: {true_ov:.2f}% \u00b1 {overlap_ci['ci_half']:.2f} (multi-crop augmentation)")
            easy["overlap_at_k"] = round(true_ov, 2)
            easy["overlap_method"] = "true_multi_crop"
            easy.setdefault("ci", {})["overlap_at_k"] = overlap_ci
            hard["overlap_at_k"] = round(true_ov, 2)
            hard["overlap_method"] = "true_multi_crop"
            hard.setdefault("ci", {})["overlap_at_k"] = overlap_ci
        else:
            print("\n  WARNING: --true-overlap set but overlap_augA/B.npy not found. Run m05c first.")

    # ── Confidence sweep ─────────────────────────────────────────────
    conf_sweep = compute_confidence_sweep(I[:, :k + 1], tags, k)
    print(f"\nConfidence sweep: {len(conf_sweep)} thresholds")

    # wandb: log all metrics
    for prefix, m in [("easy", easy), ("hard", hard)]:
        for metric in ["cycle_at_k", "overlap_at_k", "silhouette", "prec_at_k", "map_at_k", "ndcg_at_k"]:
            v = m.get(metric)
            if v is not None:
                log_metrics(wb_run, {f"{prefix}/{metric}": v})
        log_metrics(wb_run, {
            f"{prefix}/macro_prec_at_k": m["macro_avg"]["prec_at_k"],
            f"{prefix}/micro_prec_at_k": m["micro_avg"]["prec_at_k"],
        })
    for s in conf_sweep:
        if s["prec_at_k"] is not None:
            log_metrics(wb_run, {
                "confidence_sweep/threshold": s["threshold"],
                "confidence_sweep/prec_at_k": s["prec_at_k"],
                "confidence_sweep/coverage": s["coverage"],
            })

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        "encoder": args.encoder,
        "encoder_dim": enc_info["dim"],
        "easy": easy,
        "hard": hard,
        "confidence_sweep": conf_sweep,
        "k_neighbors": k,
        "num_clips": n,
        "exclusion_window_sec": EXCLUSION_WINDOW_SEC,
        "mode": "poc" if args.subset else "full",
        "subset_file": args.subset,
    }

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {metrics_file}")

    # Save kNN indices for downstream plotting (m08_plot.py)
    knn_file = enc_files["knn_indices"]
    np.save(knn_file, I[:, :k + 1])  # (N, k+1) — col 0 is self
    print(f"Saved: {knn_file} ({I[:, :k + 1].shape})")

    # ── Plots ────────────────────────────────────────────────────────
    if not args.no_plots:
        sfx = get_encoder_info(args.encoder)["suffix"]
        generate_plots(easy, hard, conf_sweep, D[:, :k + 1], k, output_dir, n, sfx=sfx)
        for plot_name in ["m06_distance_hist", "m06_confidence_sweep",
                          "m06_purity_all_keys", "m06_silhouette_per_key",
                          "m06_map_per_key", "m06_cycle_per_key",
                          "m06_map_per_value", "m06_metrics_by_key",
                          "m06_radar"]:
            log_image(wb_run, f"{plot_name}{sfx}", str(output_dir / f"{plot_name}{sfx}.png"))

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'EVALUATION SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Easy':>10} {'Hard':>10}")
    print(f"{'-'*42}")
    for m in ["cycle_at_k", "overlap_at_k", "silhouette", "prec_at_k", "map_at_k", "ndcg_at_k"]:
        ev = easy.get(m)
        hv = hard.get(m)
        label = m.replace("_", " ").title()
        fmt = ".2f" if m in ("cycle_at_k", "overlap_at_k", "prec_at_k") else ".4f"
        ev_s = f"{ev:{fmt}}" if ev is not None else "N/A"
        hv_s = f"{hv:{fmt}}" if hv is not None else "N/A"
        print(f"  {label:<18} {ev_s:>10} {hv_s:>10}")

    print(f"\n  Macro Prec@K:  Easy={easy['macro_avg']['prec_at_k']:.2f}%  Hard={hard['macro_avg']['prec_at_k']:.2f}%")
    print(f"  Micro Prec@K:  Easy={easy['micro_avg']['prec_at_k']:.2f}%  Hard={hard['micro_avg']['prec_at_k']:.2f}%")

    log_artifact(wb_run, "metrics", str(metrics_file))
    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
