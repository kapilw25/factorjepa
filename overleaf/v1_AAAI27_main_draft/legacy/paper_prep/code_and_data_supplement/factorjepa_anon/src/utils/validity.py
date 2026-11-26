"""Construct-validity statistics for a subject × metric matrix (iter19 benchmark §2b).

Pure-CPU, numpy + scipy.stats. **No `m*.py` import (rule 32):** the caller (m13, which owns
the metric registry) passes metric DIRECTIONS and FAMILY labels in. This module only does the
statistics behind a convergent/discriminant ("nomological net") study of a benchmark's metrics:

  orient_higher_better : flip 'lower'-is-better columns so every metric reads higher=better;
                         blank 'signed' columns (e.g. `order` — reported-not-ranked, §2 order-fix).
  pairwise_spearman    : null-safe Spearman corr — each pair uses only the subjects where BOTH
                         metrics are present, so partial rows never poison the matrix.
  family_summary       : reduce the corr matrix to the headline — mean WITHIN-family ρ, mean
                         BETWEEN-family ρ, their gap, and a label-permutation p-value (is the gap
                         bigger than chance family re-labellings?).

Convergent validity   = high within-family ρ  (metrics in a family move together).
Discriminant validity = lower between-family ρ (families are distinct constructs).
Criterion validity (a metric vs an EXTERNAL ground-truth capability) needs that external score,
so it lives with the caller — see the iter19 plan §2b.

USAGE (self-contained synthetic smoke — plants a 3-block matrix, asserts within >> between):
    source venv_denseworld/bin/activate
    python -u src/utils/validity.py --selftest --n-perm 2000 --seed 0
"""
import argparse

import numpy as np
from scipy.stats import rankdata

_MIN_PAIR_N = 3   # Spearman needs ≥3 shared finite points for a defined rank correlation


def orient_higher_better(matrix, directions):
    """Return a copy of `matrix` (subjects × metrics) where every column reads higher=better.

    directions[j] ∈ {'higher','lower','signed'}:
      · 'higher' → unchanged
      · 'lower'  → negated (a smaller-is-better metric now ranks higher=better)
      · 'signed' → whole column blanked to NaN (a signed diagnostic like `order` is
                   reported-not-ranked; excluding it keeps the correlation interpretable).
    Existing NaNs (missing cells) are preserved.
    """
    M = np.array(matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"matrix must be 2D (subjects x metrics), got shape {M.shape}")
    if len(directions) != M.shape[1]:
        raise ValueError(f"directions has {len(directions)} entries but matrix has "
                         f"{M.shape[1]} metric columns")
    for j, d in enumerate(directions):
        if d == "higher":
            continue
        if d == "lower":
            M[:, j] = -M[:, j]
        elif d == "signed":
            M[:, j] = np.nan
        else:
            raise ValueError(f"direction[{j}] = {d!r} — expected higher|lower|signed")
    return M


def _spearman_pair(a, b):
    """Spearman rho between two 1-D arrays over their PAIRWISE-COMPLETE rows.

    Returns (rho, n). rho is NaN when <3 shared finite points, or when either side is
    constant over the shared rows (rank correlation undefined there)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < _MIN_PAIR_N:
        return np.nan, n
    ra, rb = rankdata(a[mask]), rankdata(b[mask])   # rankdata averages ties
    if ra.std() == 0 or rb.std() == 0:
        return np.nan, n
    return float(np.corrcoef(ra, rb)[0, 1]), n


def pairwise_spearman(matrix):
    """Null-safe Spearman correlation across columns (metrics).

    matrix: subjects × metrics (NaN = missing). Returns (corr, n), each metrics × metrics.
    corr[i,j] uses only the subjects where BOTH metric i and j are finite; the diagonal is
    NaN (self-correlation carries no validity signal)."""
    M = np.asarray(matrix, float)
    m = M.shape[1]
    corr = np.full((m, m), np.nan)
    n = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(i + 1, m):
            rho, nij = _spearman_pair(M[:, i], M[:, j])
            corr[i, j] = corr[j, i] = rho
            n[i, j] = n[j, i] = nij
    return corr, n


def _within_between(corr, fam_idx):
    """Split finite off-diagonal correlations into within-family vs between-family means."""
    m = corr.shape[0]
    win, btw = [], []
    for i in range(m):
        for j in range(i + 1, m):
            if not np.isfinite(corr[i, j]):
                continue
            (win if fam_idx[i] == fam_idx[j] else btw).append(corr[i, j])
    win_mean = float(np.mean(win)) if win else np.nan
    btw_mean = float(np.mean(btw)) if btw else np.nan
    return win_mean, btw_mean, len(win), len(btw)


def family_summary(corr, families, n_perm, rng):
    """Headline convergent/discriminant numbers + a label-permutation p-value.

    corr     : metrics × metrics Spearman (from pairwise_spearman, oriented higher=better).
    families : per-metric family label (len = n_metrics).
    n_perm   : number of family-label shuffles for the permutation null.
    rng      : a numpy Generator (the caller seeds it — keeps the p-value reproducible).

    Returns dict: within, between, gap (= within − between), n_within_pairs, n_between_pairs,
    perm_p (= P[ a shuffled family-labelling reproduces a gap ≥ the observed one ]).
    """
    fams = list(families)
    if len(fams) != corr.shape[0]:
        raise ValueError(f"families has {len(fams)} entries but corr is {corr.shape[0]} wide")
    labels = sorted(set(fams))
    fam_idx = np.array([labels.index(f) for f in fams])
    win, btw, n_win, n_btw = _within_between(corr, fam_idx)
    gap = win - btw
    ge = 1   # +1: the observed labelling counts itself → never reports a literal p = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(fam_idx)
        pw, pb, _, _ = _within_between(corr, perm)
        if np.isfinite(pw) and np.isfinite(pb) and (pw - pb) >= gap:
            ge += 1
    return {"within": win, "between": btw, "gap": gap,
            "n_within_pairs": n_win, "n_between_pairs": n_btw,
            "perm_p": ge / (int(n_perm) + 1)}


def criterion_rho(metric_vals, criterion_vals, n_boot, rng):
    """Criterion validity: Spearman rho between a metric and an EXTERNAL ground-truth capability,
    over the subjects present in BOTH, with a percentile bootstrap 95% CI.

    metric_vals, criterion_vals : 1-D arrays, aligned by subject (NaN = missing). Orient the metric
    to higher=better BEFORE calling (so a positive rho always means 'the metric tracks the capability').
    n_boot : bootstrap resamples for the CI.  rng : seeded numpy Generator (reproducible CI).

    Returns dict: rho, lo, hi (2.5/97.5 percentile), n. rho/lo/hi = NaN when <4 shared finite points
    or either side is constant."""
    x = np.asarray(metric_vals, float)
    y = np.asarray(criterion_vals, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(mask.sum())
    if n < 4 or x.std() == 0 or y.std() == 0:
        return {"rho": np.nan, "lo": np.nan, "hi": np.nan, "n": n}
    rho = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if xb.std() == 0 or yb.std() == 0:
            continue
        boots.append(float(np.corrcoef(rankdata(xb), rankdata(yb))[0, 1]))
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (np.nan, np.nan)
    return {"rho": rho, "lo": lo, "hi": hi, "n": n}


def _selftest(n_perm, seed):
    """Plant a 3-family block matrix (within-family metrics share a latent construct) and
    assert the study recovers within >> between with a small permutation p."""
    rng = np.random.default_rng(seed)
    n_subj, per_fam, n_fam = 24, 4, 3
    families, cols = [], []
    for f in range(n_fam):
        latent = rng.standard_normal(n_subj)                       # the family's shared construct
        for _k in range(per_fam):
            cols.append(latent + 0.35 * rng.standard_normal(n_subj))  # noisy copies of it
            families.append(f"fam{f}")
    Mo = orient_higher_better(np.column_stack(cols), ["higher"] * len(families))
    corr, _n = pairwise_spearman(Mo)
    s = family_summary(corr, families, n_perm=n_perm, rng=rng)
    print(f"[selftest] within rho={s['within']:.3f} · between rho={s['between']:.3f} · "
          f"gap={s['gap']:.3f} · perm p={s['perm_p']:.4f} "
          f"({s['n_within_pairs']} within / {s['n_between_pairs']} between pairs)")
    assert s["within"] > s["between"], "selftest FAILED: within <= between"
    assert s["gap"] > 0.3, f"selftest FAILED: gap {s['gap']:.3f} too small for a planted 3-block matrix"
    assert s["perm_p"] < 0.05, f"selftest FAILED: perm p {s['perm_p']:.3f} not significant"
    # criterion validity: a metric that IS the capability + noise → strong positive rho, CI above 0
    cap = rng.standard_normal(40)
    cr = criterion_rho(cap + 0.3 * rng.standard_normal(40), cap, n_boot=2000, rng=rng)
    print(f"[selftest] criterion rho={cr['rho']:.3f} [{cr['lo']:.3f}, {cr['hi']:.3f}] n={cr['n']}")
    assert cr["rho"] > 0.7 and cr["lo"] > 0, "selftest FAILED: criterion rho not strongly positive"
    print("[selftest] PASS — convergent >> discriminant + criterion tracks the capability")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="construct-validity statistics (iter19 benchmark §2b)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic 3-block sanity check (no data needed)")
    ap.add_argument("--n-perm", type=int, required=True, help="family-label permutation count")
    ap.add_argument("--seed", type=int, required=True, help="RNG seed (reproducible permutation p)")
    args = ap.parse_args()
    if args.selftest:
        _selftest(args.n_perm, args.seed)
    else:
        ap.error("nothing to do — pass --selftest "
                 "(the real-matrix study runs via m13_eval_plot.plot_metric_validity)")
