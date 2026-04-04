"""Bootstrap 95% CI for retrieval metrics via scipy.stats.bootstrap (BCa method, 10K iterations).
Resample queries (clips), recompute aggregate metric. Standard IR practice (Sakai, SIGIR 2006).

Per-clip metric functions are fully vectorized (NumPy) — no Python for-loops.
115K clips × k=6: ~1s per metric (was ~10 min with Python loops)."""
import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95


def bootstrap_ci(per_query_scores: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 ci: float = CI_LEVEL, seed: int = 42) -> dict:
    """Compute bootstrap 95% CI on per-query metric scores."""
    scores = np.asarray(per_query_scores, dtype=np.float64)
    scores = scores[~np.isnan(scores)]
    n = len(scores)
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "ci_half": 0.0}

    mean = float(np.mean(scores))

    # BCa requires jackknife (N leave-one-out stats). At N=115K, the jackknife
    # array is 115K × 115K × 8 bytes ≈ 100GB → OOM on 120GB cgroup.
    # For N > 50K, use "percentile" method (jackknife-free, ~identical CI bounds
    # at this sample size — BCa correction is negligible when N >> 1000).
    method = "BCa" if n <= 50_000 else "percentile"

    result = scipy_bootstrap(
        (scores,),
        statistic=np.mean,
        n_resamples=n_boot,
        confidence_level=ci,
        method=method,
        random_state=np.random.default_rng(seed),
    )
    ci_lo = float(result.confidence_interval.low)
    ci_hi = float(result.confidence_interval.high)

    return {
        "mean": round(mean, 6),
        "ci_lo": round(ci_lo, 6),
        "ci_hi": round(ci_hi, 6),
        "ci_half": round((ci_hi - ci_lo) / 2, 6),
    }


# ═══════════════════════════════════════════════════════════════════════
# Tag label pre-encoding: convert list-of-dicts to NumPy int arrays ONCE
# ═══════════════════════════════════════════════════════════════════════

def encode_tags_to_labels(tags: list, fields: list) -> dict:
    """Convert tags list-of-dicts to {field: int_labels_array}.

    Each unique string value gets a unique int. "unknown"/missing → -1.
    Call ONCE before all per-clip metric functions.
    Uses pd.factorize for vectorized encoding when available, falls back to dict loop.

    Returns:
        {field_name: np.ndarray of shape (n_clips,) with int labels}
    """
    n = len(tags)
    label_arrays = {}

    try:
        import pandas as pd
        for field in fields:
            values = [t.get(field) for t in tags]
            series = pd.Series(values)
            # factorize: unique string → int, NaN/None → -1
            codes, _ = pd.factorize(series)
            labels = codes.astype(np.int32)
            # Mark "unknown" as -1 too
            unknown_mask = series == "unknown"
            labels[unknown_mask] = -1
            label_arrays[field] = labels
    except ImportError:
        # Fallback: dict-based encoding
        for field in fields:
            vocab = {}
            next_id = 0
            labels = np.full(n, -1, dtype=np.int32)
            for i, t in enumerate(tags):
                val = t.get(field)
                if val is None or val == "unknown":
                    continue
                if val not in vocab:
                    vocab[val] = next_id
                    next_id += 1
                labels[i] = vocab[val]
            label_arrays[field] = labels

    return label_arrays


# ═══════════════════════════════════════════════════════════════════════
# Vectorized per-clip metrics (NumPy, no Python loops)
# ═══════════════════════════════════════════════════════════════════════

def per_clip_prec_at_k(indices: np.ndarray, tags, k: int,
                        field: str = "scene_type",
                        label_arrays: dict = None) -> np.ndarray:
    """Per-clip Prec@K — vectorized. Accepts pre-encoded label_arrays or raw tags."""
    n = indices.shape[0]
    neighbors = indices[:, 1:k + 1]  # (N, k) — skip self at col 0

    if label_arrays is not None and field in label_arrays:
        labels = label_arrays[field]
    else:
        labels = encode_tags_to_labels(tags, [field])[field]

    query_labels = labels[:n]                          # (N,)
    neighbor_labels = labels[np.clip(neighbors, 0, len(labels) - 1)]  # (N, k)

    # Mask invalid indices
    valid = (neighbors >= 0) & (neighbors < len(labels))
    matches = (neighbor_labels == query_labels[:, None]) & valid  # (N, k)

    hits = matches.sum(axis=1).astype(np.float64)
    totals = valid.sum(axis=1).astype(np.float64)
    scores = np.where(totals > 0, hits / totals * 100, 0.0)
    return scores


def per_clip_map_at_k(indices: np.ndarray, tags, k: int,
                       field: str = "scene_type",
                       label_arrays: dict = None) -> np.ndarray:
    """Per-clip AP@K — vectorized."""
    n = indices.shape[0]
    neighbors = indices[:, 1:k + 1]  # (N, k)

    if label_arrays is not None and field in label_arrays:
        labels = label_arrays[field]
    else:
        labels = encode_tags_to_labels(tags, [field])[field]

    query_labels = labels[:n]
    neighbor_labels = labels[np.clip(neighbors, 0, len(labels) - 1)]

    valid = (neighbors >= 0) & (neighbors < len(labels))
    matches = (neighbor_labels == query_labels[:, None]) & valid  # (N, k) bool

    # Cumulative hits and precision@rank
    cum_hits = np.cumsum(matches.astype(np.float64), axis=1)  # (N, k)
    ranks = np.arange(1, k + 1, dtype=np.float64)             # (k,)
    precision_at_rank = cum_hits / ranks                        # (N, k)

    # AP = sum(precision_at_rank * is_relevant) / k
    ap = (precision_at_rank * matches).sum(axis=1) / k
    return ap


def per_clip_cycle_at_k(indices: np.ndarray, k: int) -> np.ndarray:
    """Per-clip Cycle@K — vectorized."""
    n = indices.shape[0]
    nearest = indices[:, 1]  # (N,) — nearest neighbor of each clip

    # For each clip i, check if i appears in nearest[i]'s neighbor list
    valid = (nearest >= 0) & (nearest < n)
    scores = np.full(n, np.nan)

    # Get neighbor lists of each clip's nearest neighbor
    valid_idx = np.where(valid)[0]
    nearest_valid = nearest[valid_idx]

    # For each valid clip, check if query is in its nearest neighbor's top-k
    nn_neighbors = indices[nearest_valid, 1:k + 1]  # (n_valid, k)
    # Check if query index appears in any column
    query_in_nn = (nn_neighbors == valid_idx[:, None]).any(axis=1)
    scores[valid_idx] = query_in_nn.astype(np.float64)

    return scores


def per_clip_ndcg_at_k(indices: np.ndarray, tags, k: int,
                        taxonomy: dict,
                        label_arrays: dict = None) -> np.ndarray:
    """Per-clip nDCG@K — vectorized."""
    single_fields = [f for f, spec in taxonomy.items() if spec["type"] == "single"]
    n_fields = len(single_fields)
    n = indices.shape[0]
    n_tags = len(tags) if isinstance(tags, list) else tags

    if label_arrays is None:
        label_arrays = encode_tags_to_labels(tags, single_fields)

    neighbors = indices[:, 1:k + 1]  # (N, k)
    valid = (neighbors >= 0) & (neighbors < (n_tags if isinstance(n_tags, int) else len(tags)))
    safe_neighbors = np.clip(neighbors, 0, (n_tags if isinstance(n_tags, int) else len(tags)) - 1)

    # Count matching fields across all single_fields
    # grades[i, r] = (# matching fields between clip i and neighbor at rank r) / n_fields
    grades = np.zeros((n, k), dtype=np.float64)
    for field in single_fields:
        labels = label_arrays[field]
        query_labels = labels[:n]                          # (N,)
        neighbor_labels = labels[safe_neighbors]           # (N, k)
        field_match = (neighbor_labels == query_labels[:, None]) & (query_labels[:, None] != -1)
        grades += field_match.astype(np.float64)
    grades /= n_fields
    grades *= valid  # zero out invalid neighbors

    # DCG = sum(grade / log2(rank + 2)) for rank 0..k-1
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))  # (k,)
    dcg = (grades * discount).sum(axis=1)  # (N,)

    # IDCG = DCG with grades sorted descending
    ideal_grades = np.sort(grades, axis=1)[:, ::-1]
    idcg = (ideal_grades * discount).sum(axis=1)

    scores = np.where(idcg > 0, dcg / idcg, np.nan)
    return scores
