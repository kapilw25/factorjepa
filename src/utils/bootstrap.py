"""Bootstrap 95% CI for retrieval metrics via scipy.stats.bootstrap (BCa method, 10K iterations).
Resample queries (clips), recompute aggregate metric. Standard IR practice (Sakai, SIGIR 2006)."""
import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95


def bootstrap_ci(per_query_scores: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 ci: float = CI_LEVEL, seed: int = 42) -> dict:
    """Compute bootstrap BCa 95% CI on per-query metric scores.

    Uses scipy.stats.bootstrap (BCa = bias-corrected and accelerated).
    Resamples queries (clips), not query-neighbor pairs.

    Args:
        per_query_scores: 1D array of per-clip metric values.
        n_boot: Bootstrap iterations (default 10,000 for publication quality).
        ci: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        {"mean": float, "ci_lo": float, "ci_hi": float, "ci_half": float}
    """
    scores = np.asarray(per_query_scores, dtype=np.float64)
    scores = scores[~np.isnan(scores)]
    n = len(scores)
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "ci_half": 0.0}

    mean = float(np.mean(scores))

    # scipy.stats.bootstrap requires tuple of (array,)
    result = scipy_bootstrap(
        (scores,),
        statistic=np.mean,
        n_resamples=n_boot,
        confidence_level=ci,
        method="BCa",
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


def per_clip_prec_at_k(indices: np.ndarray, tags: list, k: int,
                        field: str = "scene_type") -> np.ndarray:
    """Per-clip Prec@K scores (for bootstrap input)."""
    n = indices.shape[0]
    n_tags = len(tags)
    scores = np.zeros(n)

    for i in range(n):
        my_val = tags[i].get(field, "unknown")
        hits = 0
        total = 0
        for j in indices[i, 1:k + 1]:
            if 0 <= j < n_tags:
                if tags[j].get(field, "unknown") == my_val:
                    hits += 1
                total += 1
        scores[i] = (hits / total * 100) if total > 0 else 0.0

    return scores


def per_clip_map_at_k(indices: np.ndarray, tags: list, k: int,
                       field: str = "scene_type") -> np.ndarray:
    """Per-clip AP@K scores (for bootstrap input)."""
    n = indices.shape[0]
    n_tags = len(tags)
    scores = np.zeros(n)

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
        scores[i] = ap_sum / k

    return scores


def per_clip_cycle_at_k(indices: np.ndarray, k: int) -> np.ndarray:
    """Per-clip Cycle@K (1 if reciprocal, 0 if not, NaN if invalid)."""
    n = indices.shape[0]
    scores = np.full(n, np.nan)

    for i in range(n):
        nearest = indices[i, 1]
        if nearest < 0 or nearest >= n:
            continue
        scores[i] = 1.0 if i in set(indices[nearest, 1:k + 1].tolist()) else 0.0

    return scores


def per_clip_ndcg_at_k(indices: np.ndarray, tags: list, k: int,
                        taxonomy: dict) -> np.ndarray:
    """Per-clip nDCG@K scores (for bootstrap input)."""
    single_fields = [f for f, spec in taxonomy.items() if spec["type"] == "single"]
    n_fields = len(single_fields)
    n = indices.shape[0]
    n_tags = len(tags)
    scores = np.full(n, np.nan)

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
        scores[i] = dcg / idcg if idcg > 0 else np.nan

    return scores
