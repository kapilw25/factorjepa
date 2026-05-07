"""
CPU-only multi-encoder comparison: grouped bar chart, spatial-temporal tradeoff scatter,
combined radar, LaTeX table with 95% CI. Reads m06_metrics_*.json + m06b_temporal_corr_*.json.

USAGE:
    python -u src/m08b_compare.py --SANITY 2>&1 | tee logs/m08b_compare_sanity.log
    python -u src/m08b_compare.py --POC --subset data/subset_10k.json 2>&1 | tee logs/m08b_compare_poc.log
    python -u src/m08b_compare.py --FULL 2>&1 | tee logs/m08b_compare_full.log

═════════════════════════════════════════════════════════════════════════
DELETE-PROTECTION POLICY — m08b ALWAYS RECOMPUTES (no `--cache-policy`).
═════════════════════════════════════════════════════════════════════════
This script does NOT route through `utils.cache_policy` (no `add_cache_policy_arg`,
no `guarded_delete`, no `wipe_output_dir`). DO NOT re-add cache_policy gating
to this script in the future. Reasons:

  1. m08b is a PURE VISUALIZATION FUNCTION of (m05 .npy + m06 .json + tags.json).
     Same inputs → same outputs. There is NO expensive intermediate state worth
     preserving (no GPU model load, no checkpoint, no resume cursor). Wall-time
     is ~30s CPU at full eval scale (308 clips × N encoders).

  2. m08b OUTPUTS ARE READ-ONLY ARTIFACTS — they are PNG/PDF plots, a .tex table,
     and a paired_bootstrap_results.json that downstream stages NEVER consume.
     Their only audience is humans (papers, PRs, decision tables). So the failure
     mode of preserving stale plots is "wrong claim in research artifact",
     which is *worse* than the cost of recomputing.

  3. The 2026-04-27 INCIDENT (errors_N_fixes #80): `outputs/full/m08b_compare/`
     held 2-encoder PNGs from an Apr 25 run; today's 4-variant `run_eval.sh`
     wrote NEW per-variant `outputs/full/<variant>/eval/m08b_*.png` but the stale
     dir remained, masking the missing aggregate-4-encoder plots from a casual
     `ls outputs/full/m08b_compare/` inspection. Cache-policy=1/keep was the
     enabler — Option A removed it entirely so this class of bug cannot recur.

  4. CLAUDE.md GPU PIPELINE CHECKLIST item (3) ("add_cache_policy_arg + interactive
     prompt") gates DESTRUCTIVE deletes of expensive caches (m05 frozen embeds,
     m09 training checkpoints, etc.). m08b has nothing destructive to delete —
     its `output_dir` contains only its own write-once outputs that get re-written
     this very run. The checklist item is intentionally bypassed here; this is
     the FIRST documented exception.

  5. Adding cache_policy back would also re-introduce a `args.cache_policy` access
     site — i.e. one more place where the F821 silent-fallback class (errors_N_fixes
     #79) could regress.

  Behavioral spec post-removal:
    - On every invocation, m08b WIPES `--output-dir` (shutil.rmtree + mkdir empty)
      before writing its 8 plots + tex + paired_bootstrap_results.json.
    - No `--cache-policy` CLI arg. Orchestrators (`scripts/legacy2/run_eval.sh`) MUST NOT
      pass `--cache-policy` to m08b — it will fail-loud with `unrecognized arg`.
    - Stale-radar (n<3) PURGE is unconditional (was guarded_delete before).
═════════════════════════════════════════════════════════════════════════
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.progress import make_pbar
from utils.config import (
    add_subset_arg, get_output_dir, get_module_output_dir,
    get_encoder_files, get_encoder_info,
)
from utils.wandb_utils import add_wandb_args, init_wandb, log_image, finish_wandb
# DO NOT import utils.cache_policy here. m08b is exempt — see module docstring.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Encoder display config ───────────────────────────────────────────

ENCODER_LABELS = {
    "vjepa": "V-JEPA 2\n(frozen, 1408d)",
    "random": "Random\n(1408d)",
    "dinov2": "DINOv2\n(ViT-L, 1024d)",
    "clip": "CLIP\n(ViT-L, 768d)",
    "vjepa_shuffled": "Shuffled\nV-JEPA (frozen)",
    "vjepa_adapted": "V-JEPA 2\n(adapted, 1408d)",
    "vjepa_surgical": "V-JEPA 2\n(surgical, 1408d)",
}
ENCODER_COLORS = {
    "vjepa": "#2196F3",
    "random": "#9E9E9E",
    "dinov2": "#4CAF50",
    "clip": "#FF9800",
    "vjepa_shuffled": "#E91E63",
    "vjepa_adapted": "#7B1FA2",
    "vjepa_surgical": "#00695C",
}
# Display order: V-JEPA first (main), then baselines
ENCODER_ORDER = ["vjepa", "random", "dinov2", "clip", "vjepa_shuffled"]

METRICS_DISPLAY = [
    ("cycle_at_k", "Cycle@K (%)"),
    ("dim_consistency_at_k", "DimConsist@K (%)"),
    ("prec_at_k", "Prec@K (%)"),
    ("map_at_k", "mAP@K"),
    ("ndcg_at_k", "nDCG@K"),
    ("silhouette", "Silhouette"),
]


# ── Load all available encoder metrics ───────────────────────────────

def load_all_metrics(output_dir: Path, encoder_list: list = None) -> dict:
    """Load m06_metrics_*.json for specified or all registered encoders."""
    results = {}
    names = encoder_list if encoder_list else ENCODER_ORDER
    for name in names:
        files = get_encoder_files(name, output_dir)
        if files["metrics"].exists():
            with open(files["metrics"]) as f:
                results[name] = json.load(f)
            print(f"  Loaded: {name} ({files['metrics'].name})")
    return results


def compute_paired_bootstrap(metrics_dir: Path, output_dir: Path,
                             encoder_list: list = None) -> None:
    """Paired-bootstrap BCa CI on per-clip (surgical_i − frozen) deltas for ALL surgicals.

    iter10 Option C, extended 2026-04-27 (errors_N_fixes #80 follow-up): aggregate-mode
    run_eval.sh passes multiple surgical encoders (e.g. surgical_noDI, surgical_3stage_DI,
    surgical_3stage_DI_multitask) — this function now loops over EVERY encoder containing
    "surgical" in its name and pairs each against the single frozen baseline.

    Requires m06 to have written per_clip_{encoder}_{mode}.npz for the frozen encoder
    and each surgical encoder under metrics_dir. Writes paired_bootstrap_results.json
    with `comparisons: [{surgical, modes}, ...]` — one entry per surgical. Per-mode
    schema unchanged (frozen_mean / surgical_mean / delta_mean / delta_ci_* /
    p_value_vs_zero / n_valid). Silent no-op when per-clip files are missing for a
    given (surgical, mode) pair (back-compat with pre-iter10 m06 runs).

    Headline 🏆 line picks the BEST surgical by Easy Prec@K Δ — publishable claim.
    Multi-surgical ranking table follows for at-a-glance technique comparison.
    """
    from utils.bootstrap import paired_bca

    names = encoder_list if encoder_list else ENCODER_ORDER
    frozen_name = next((n for n in names if "frozen" in n.lower()), None)
    # Widened 2026-04-27 (errors_N_fixes #80 follow-up): match every adapted-from-frozen
    # encoder, not just "surgical". Includes ExPLoRA + future "_adapted" variants so each
    # gets its own paired Δ vs frozen. Baseline encoders (random/oracle/dinov2/clip/
    # vjepa_shuffled) are intentionally EXCLUDED — they are different model families,
    # paired-bootstrap-vs-frozen would answer "is V-JEPA better than DINOv2" which is a
    # different research question than "did our adaptation move the needle".
    ADAPTED_KEYWORDS = ("surgical", "explora", "adapted")
    adapted_names = [n for n in names
                     if any(k in n.lower() for k in ADAPTED_KEYWORDS)]
    if frozen_name is None or not adapted_names:
        print(f"\n[paired-bootstrap] skip: need BOTH a frozen and ≥1 adapted encoder "
              f"(name contains one of {ADAPTED_KEYWORDS}), "
              f"got frozen={frozen_name} adapted={adapted_names}")
        return

    metrics_list = ["prec_at_k", "map_at_k", "cycle_at_k", "ndcg_at_k"]
    results = {"frozen": frozen_name, "comparisons": []}
    headline_pool = []  # (adapted_name, easy_prec_delta_mean) — for 🏆 selection

    for adapted_name in adapted_names:
        print(f"\n{'='*70}")
        print(f"PAIRED BOOTSTRAP (BCa 95 % CI, iter10 Option C) — {adapted_name} − {frozen_name}")
        print(f"{'='*70}")

        comp = {"adapted": adapted_name, "modes": {}}
        any_mode_computed = False
        for mode in ["easy", "hard"]:
            frozen_npz = metrics_dir / f"per_clip_{frozen_name}_{mode}.npz"
            adapted_npz = metrics_dir / f"per_clip_{adapted_name}_{mode}.npz"
            if not (frozen_npz.exists() and adapted_npz.exists()):
                print(f"  [paired-bootstrap] skip {mode}: per-clip .npz missing "
                      f"(frozen={frozen_npz.exists()}, adapted={adapted_npz.exists()}). "
                      f"Re-run m06 with iter10 patch to produce per_clip_*.npz.")
                continue
            f_data = np.load(frozen_npz, allow_pickle=True)
            a_data = np.load(adapted_npz, allow_pickle=True)
            f_keys = list(f_data["clip_keys"])
            a_keys = list(a_data["clip_keys"])
            # Intersect key sets: m05 writes embeddings in decode-completion order which
            # is non-deterministic across parallel-worker races, and partial-tolerance
            # (stuck_clips path, #77) means frozen and adapted runs may have different
            # failed subsets. Paired bootstrap requires identical per-clip order, so build
            # the intersection once and reindex every per-metric array in both .npz files
            # to that common order.
            common = sorted(set(f_keys) & set(a_keys))
            if len(common) == 0:
                print(f"FATAL: zero-overlap clip_keys between {frozen_npz.name} and "
                      f"{adapted_npz.name} (f={len(f_keys)}, a={len(a_keys)}). "
                      f"Cannot run paired bootstrap.")
                sys.exit(1)
            f_idx = {k: i for i, k in enumerate(f_keys)}
            a_idx = {k: i for i, k in enumerate(a_keys)}
            f_order = np.array([f_idx[k] for k in common], dtype=np.int64)
            a_order = np.array([a_idx[k] for k in common], dtype=np.int64)
            n_dropped_f = len(f_keys) - len(common)
            n_dropped_a = len(a_keys) - len(common)
            if f_keys != a_keys:
                print(f"  [paired-bootstrap] key-set alignment: frozen={len(f_keys)}, "
                      f"adapted={len(a_keys)}, common={len(common)} "
                      f"(dropped frozen-only={n_dropped_f}, adapted-only={n_dropped_a}) — "
                      f"reindexed both to common order before paired BCa")

            mode_results = {
                "n_clips": len(common),
                "n_frozen_only_dropped": n_dropped_f,
                "n_adapted_only_dropped": n_dropped_a,
                "metrics": {},
            }
            header = f"{'Metric':<14s} {'Frozen':>10s} {'Adapted':>10s} {'Δ':>10s} {'CI_half':>10s} {'p_vs_0':>10s}"
            print(f"\n--- {mode.upper()} ---")
            print(header)
            print("-" * len(header))
            for metric in metrics_list:
                f_arr = np.asarray(f_data[metric], dtype=np.float64)[f_order]
                a_arr = np.asarray(a_data[metric], dtype=np.float64)[a_order]
                valid = ~(np.isnan(f_arr) | np.isnan(a_arr))
                deltas = a_arr[valid] - f_arr[valid]
                paired = paired_bca(deltas)
                mode_results["metrics"][metric] = {
                    "frozen_mean": float(np.mean(f_arr[valid])),
                    "adapted_mean": float(np.mean(a_arr[valid])),
                    "delta_mean": paired["mean"],
                    "delta_ci_lo": paired["ci_lo"],
                    "delta_ci_hi": paired["ci_hi"],
                    "delta_ci_half": paired["ci_half"],
                    "p_value_vs_zero": paired["p_value_vs_zero"],
                    "n_valid": paired["n"],
                }
                print(f"{metric:<14s} {np.mean(f_arr[valid]):>10.4f} "
                      f"{np.mean(a_arr[valid]):>10.4f} {paired['mean']:>+10.4f} "
                      f"{paired['ci_half']:>10.4f} {paired['p_value_vs_zero']:>10.4f}")
            comp["modes"][mode] = mode_results
            any_mode_computed = True

        if any_mode_computed:
            results["comparisons"].append(comp)
            if "easy" in comp["modes"]:
                headline_pool.append(
                    (adapted_name, comp["modes"]["easy"]["metrics"]["prec_at_k"]["delta_mean"])
                )

    if not results["comparisons"]:
        return

    out = output_dir / "paired_bootstrap_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    if not headline_pool:
        return

    # Headline: BEST adapted encoder by Easy Prec@K Δ — publishable claim.
    headline_pool_sorted = sorted(headline_pool, key=lambda x: -x[1])
    best_name, best_delta = headline_pool_sorted[0]
    best_comp = next(c for c in results["comparisons"] if c["adapted"] == best_name)
    pk = best_comp["modes"]["easy"]["metrics"]["prec_at_k"]
    verdict = ("✅ Δ significant (CI excludes 0)"
               if pk["delta_ci_lo"] > 0 or pk["delta_ci_hi"] < 0
               else "🟡 Δ not significant (CI straddles 0)")
    print(f"\n🏆 BEST adapted encoder (by Easy Prec@K Δ): {best_name}")
    print(f"   Paired Prec@K (Easy, N={best_comp['modes']['easy']['n_clips']}): "
          f"Δ = {pk['delta_mean']:+.4f} ± {pk['delta_ci_half']:.4f} "
          f"(95% CI [{pk['delta_ci_lo']:+.4f}, {pk['delta_ci_hi']:+.4f}], "
          f"p={pk['p_value_vs_zero']:.4f}) {verdict}")

    # Multi-adapted ranking table — at-a-glance technique comparison.
    if len(headline_pool) > 1:
        print(f"\nAll adapted encoder comparisons vs {frozen_name} "
              f"(Easy Prec@K Δ, ranked best-first):")
        print(f"  {'Adapted encoder':<48s} {'Δ Prec@K':>10s} {'p_vs_0':>10s}  Verdict")
        for sname, _ in headline_pool_sorted:
            spk = next(c for c in results["comparisons"]
                       if c["adapted"] == sname)["modes"]["easy"]["metrics"]["prec_at_k"]
            sv = ("✅" if spk["delta_ci_lo"] > 0 or spk["delta_ci_hi"] < 0 else "🟡")
            print(f"  {sname:<48s} {spk['delta_mean']:>+10.4f} {spk['p_value_vs_zero']:>10.4f}  {sv}")


def create_paired_delta_chart(output_dir: Path) -> None:
    """Δ-vs-frozen bar chart with paired-bootstrap 95 % CI per adapted encoder.

    Reads {output_dir}/paired_bootstrap_results.json (written by compute_paired_bootstrap).
    Solves the "all encoders look identical" visualization gotcha: the standard
    encoder_comparison plot uses INDEPENDENT 95 % CIs (~±3.5 pp at N=308), which
    visually overlap and hide the real ~±0.6 pp paired CIs that actually catch
    significance. This Δ chart shows ONLY paired CIs centered on Δ=0 — bars
    excluding 0 are statistically significant gains over frozen.

    Layout: 2 rows (Easy/Hard) × 4 cols (prec, mAP, cycle, nDCG). Bars per
    adapted encoder, height = Δ vs frozen, error = paired CI half. Color: green
    (✅) if CI excludes 0; gray (🟡) if straddles 0. Horizontal red dashed line at
    Δ=0 for "no improvement" reference.
    """
    pbr_path = output_dir / "paired_bootstrap_results.json"
    if not pbr_path.exists():
        print(f"  [m08b] skip paired-Δ chart: {pbr_path.name} not found")
        return
    pbr = json.load(open(pbr_path))
    comps = pbr.get("comparisons", [])
    if not comps:
        print(f"  [m08b] skip paired-Δ chart: no comparisons in {pbr_path.name}")
        return

    # Per-metric display scale: m06 stores Prec@K in percent (0–100), but
    # mAP@K / Cycle@K / nDCG@K as fractions (0–1). To put Cycle@K Δ on the
    # same pp axis as Prec@K, multiply its fraction-Δ by 100. mAP and nDCG
    # stay as fractions (conventional reporting).
    metrics = [("prec_at_k",  "Prec@K Δ (pp)",  1.0),    # already pp
               ("map_at_k",   "mAP@K Δ",        1.0),    # fraction
               ("cycle_at_k", "Cycle@K Δ (pp)", 100.0),  # fraction → pp
               ("ndcg_at_k",  "nDCG@K Δ",       1.0)]    # fraction
    modes = ["easy", "hard"]
    encoder_short = [c["adapted"].replace("vjepa_2_1_", "") for c in comps]

    fig, axes = plt.subplots(len(modes), len(metrics),
                             figsize=(3.6 * len(metrics), 3.4 * len(modes)),
                             squeeze=False)
    fig.suptitle(f"Paired-bootstrap Δ vs {pbr['frozen']} (BCa 95 % CI · "
                 f"green = CI excludes 0 / gray = straddles 0 · '*' = significant)",
                 fontsize=11, fontweight="bold", y=0.995)

    for r, mode in enumerate(modes):
        for col, (mkey, mlabel, scale) in enumerate(metrics):
            ax = axes[r][col]
            x = np.arange(len(comps))
            vals, errs, colors = [], [], []
            for c in comps:
                if mode not in c["modes"] or mkey not in c["modes"][mode]["metrics"]:
                    vals.append(0.0); errs.append(0.0); colors.append("#cccccc")
                    continue
                pk = c["modes"][mode]["metrics"][mkey]
                vals.append(pk["delta_mean"] * scale)
                errs.append(pk["delta_ci_half"] * scale)
                sig = pk["delta_ci_lo"] > 0 or pk["delta_ci_hi"] < 0
                colors.append("#2E7D32" if sig else "#888888")  # green=sig, gray=ns
            ax.bar(x, vals, color=colors, alpha=0.85, yerr=errs,
                   capsize=4, error_kw={"lw": 1.2, "ecolor": "#222"})
            ax.axhline(0.0, color="#C62828", linestyle="--", alpha=0.6, linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(encoder_short, fontsize=7, rotation=20, ha="right")
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(f"{mlabel} — {mode.capitalize()}",
                         fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.25, linewidth=0.6)
            # Annotate bar tops with Δ value + ASCII significance marker
            # ('*' = CI excludes 0; DejaVu Sans lacks ✅/✓ glyphs).
            for xi, v, e in zip(x, vals, errs):
                marker = "*" if e > 0 and abs(v) > e else ""
                ax.text(xi, v + (e if v >= 0 else -e),
                        f"{v:+.3f}{marker}", ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=7, color="#222")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in (".png", ".pdf"):
        out = output_dir / f"m08b_paired_delta{ext}"
        plt.savefig(out, dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'm08b_paired_delta.png'} + .pdf")


def load_all_temporal(output_dir: Path, encoder_list: list = None) -> dict:
    """Load m06b_temporal_corr_*.json for specified or all registered encoders."""
    results = {}
    names = encoder_list if encoder_list else ENCODER_ORDER
    for name in names:
        sfx = get_encoder_info(name)["suffix"]
        temporal_file = output_dir / f"m06b_temporal_corr{sfx}.json"
        if temporal_file.exists():
            with open(temporal_file) as f:
                results[name] = json.load(f)
            print(f"  Loaded temporal: {name} ({temporal_file.name})")
    return results


TEMPORAL_METRICS_DISPLAY = [
    ("spearman_rho", "Spearman rho"),
    ("temporal_prec_at_k", "Temp Prec@K (%)"),
    ("motion_retrieval_map", "Motion mAP"),
]


# ── Terminal Summary Table ───────────────────────────────────────────

def print_summary_table(all_metrics: dict):
    """Pretty-print comparison table to terminal."""
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if not encoders:
        print("No encoder metrics found.")
        return

    # Header
    name_w = 18
    col_w = 12
    print(f"\n{'='*80}")
    print(f"{'ENCODER COMPARISON':^80}")
    print(f"{'='*80}")

    header = f"{'Encoder':<{name_w}} {'Dim':>5}"
    for _, label in METRICS_DISPLAY:
        header += f" {label:>{col_w}}"
    print(header)
    print("-" * len(header))

    for mode in ["easy", "hard"]:
        if mode == "hard":
            print("\n--- Hard Mode ---")
        for enc in encoders:
            m = all_metrics[enc].get(mode, {})
            dim = all_metrics[enc].get("encoder_dim", get_encoder_info(enc)["dim"])
            row = f"{enc:<{name_w}} {dim:>5}"
            for key, _ in METRICS_DISPLAY:
                val = m.get(key)
                if val is None:
                    row += f" {'N/A':>{col_w}}"
                elif key in ("cycle_at_k", "dim_consistency_at_k", "prec_at_k"):
                    row += f" {val:>{col_w}.2f}"
                else:
                    row += f" {val:>{col_w}.4f}"
            print(row)

        if mode == "easy":
            print("\n--- Easy Mode (above) ---")


# ── Grouped Bar Chart ────────────────────────────────────────────────

def create_bar_chart(all_metrics: dict, output_dir: Path):
    """Per-metric encoder comparison — Easy mode only.

    Why Easy only (#78, 2026-04-21): the paper's decision gate uses `m['easy']['prec_at_k']`
    (run_iter9_10k.sh:169-187). Hard mode is a separate robustness probe (temporal-locality
    exclusion), useful for the ablations table but not for the Surgery > Frozen Δ ≥ 3 pp headline.
    Combining easy+hard hides directional regressions in either side — see chat thread 2026-04-21
    for full pros/cons.

    Y-axis is auto-scaled to [min-CI-pad, max+CI-pad] so noise-level deltas + CI bars on
    Cycle@K / nDCG@K are visible (zero-based wasted most of the frame).
    """
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2:
        print("Need at least 2 encoders for comparison chart.")
        return

    metrics_to_plot = [(k, l) for k, l in METRICS_DISPLAY if k != "silhouette"]
    n_metrics = len(metrics_to_plot)
    n_encoders = len(encoders)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), squeeze=False)
    axes = axes[0]

    for ax_idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        x = np.arange(n_encoders)
        easy_vals = []
        easy_errs = []
        colors = []

        for enc in encoders:
            easy_v = all_metrics[enc].get("easy", {}).get(metric_key)
            easy_vals.append(easy_v if easy_v is not None else 0)
            colors.append(ENCODER_COLORS.get(enc, "#888"))
            easy_ci = all_metrics[enc].get("easy", {}).get("ci", {}).get(metric_key, {})
            easy_errs.append(easy_ci.get("ci_half", 0))

        ax.bar(x, easy_vals, 0.6, color=colors, alpha=0.85,
               yerr=easy_errs, capsize=4, error_kw={"lw": 1.2, "ecolor": "#222"})

        # Auto-scaled y-limit: tight around data ± CI, with 15% padding on each side so
        # bars don't clip and error caps + value labels have breathing room. Floor at 0
        # (percentages can't go negative; single-bar metrics otherwise look inverted).
        vals_arr = np.array(easy_vals, dtype=float)
        errs_arr = np.array(easy_errs, dtype=float)
        lo = max(0.0, float((vals_arr - errs_arr).min()))
        hi = float((vals_arr + errs_arr).max())
        pad = max(0.15 * (hi - lo), 0.02 * hi) if hi > 0 else 1
        ax.set_ylim(max(0.0, lo - pad), hi + pad)

        # Value labels above each bar for quick-read at noise-level deltas.
        for xi, v, e in zip(x, easy_vals, easy_errs):
            ax.text(xi, v + e + pad * 0.1, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#222")

        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([enc.replace("_", "\n") for enc in encoders],
                           fontsize=8, rotation=0)
        ax.tick_params(axis="y", labelsize=9)

    _n_eval = all_metrics[encoders[0]].get("num_clips", "")
    _eval_str = f", N={_n_eval:,} eval clips" if _n_eval else ""
    plt.suptitle(f"Encoder Comparison (Easy mode — paper gate · 95 % CI{_eval_str})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_encoder_comparison{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_encoder_comparison.png'}")


# ── Radar Plot (Spatial + Temporal hemispheres) ─────────────────────

def create_radar_plot(all_metrics: dict, output_dir: Path,
                      all_temporal: dict = None):
    """Combined radar: 5 spatial axes (left) + 3 temporal axes (right).

    Spatial from m06, temporal from m06b. Hemisphere shading shows the
    spatial-temporal tradeoff at a glance: DINOv2 bulges left, V-JEPA
    bulges right, shuffled loses right side, random stays near center.
    """
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2:
        return

    # 5 spatial + 3 temporal axes (ordered so spatial=left, temporal=right)
    # #77: dim_consistency_at_k is NOT produced by m06 (no upstream source in any
    # src/m*.py — verified 2026-04-21). Including it here created a "ghost axis" that
    # hit the "all encoders equal" fallback at line ~294 (→ both plotted at 100) and
    # misled readers into thinking the metric existed. Dropped until m06 actually
    # computes it. Restore this line when dim-consistency lands in m06.
    spatial_keys = ["prec_at_k", "map_at_k", "cycle_at_k", "ndcg_at_k"]
    spatial_labels = ["Prec@K", "mAP@K", "Cycle@K", "nDCG@K"]

    temporal_keys = ["spearman_rho", "temporal_prec_at_k", "temporal_locality_inv"]
    temporal_labels = ["Spearman \u03c1", "Temp Prec@K", "Temp Locality"]

    has_temporal = all_temporal and any(
        all_temporal.get(e, {}).get("spearman_rho") is not None for e in encoders)

    if has_temporal:
        all_keys = spatial_keys + temporal_keys
        all_labels = spatial_labels + temporal_labels
    else:
        all_keys = spatial_keys
        all_labels = spatial_labels

    n_metrics = len(all_keys)
    n_spatial = len(spatial_keys)

    # Collect raw values
    raw = {}
    for enc in encoders:
        easy = all_metrics[enc].get("easy", {})
        vals = []
        for k in all_keys:
            if k in spatial_keys:
                vals.append(easy.get(k))
            elif k == "temporal_locality_inv":
                # Invert locality ratio: lower=better, so plot 1-ratio
                ratio = (all_temporal or {}).get(enc, {}).get(
                    "temporal_locality", {}).get("ratio")
                vals.append(1.0 - ratio if ratio is not None else None)
            else:
                vals.append((all_temporal or {}).get(enc, {}).get(k))
        raw[enc] = vals

    # Per-axis min-max normalization to [0, 100].
    # Max-only normalization (v/max*100) clusters values near the outer edge when
    # all encoders score similarly (e.g., nDCG 92-96% all map to 96-100).
    # Min-max spreads them: worst encoder → 0, best → 100, differences visible.
    mins = []
    maxes = []
    for i in range(n_metrics):
        axis_vals = [raw[e][i] for e in encoders if raw[e][i] is not None]
        mins.append(min(axis_vals) if axis_vals else 0)
        maxes.append(max(axis_vals) if axis_vals else 1)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

    # Hemisphere shading (spatial left, temporal right)
    if has_temporal:
        spatial_start = angles[0]
        spatial_end = angles[n_spatial - 1]
        temporal_start = angles[n_spatial]
        temporal_end = angles[-2]
        ax.fill_between(
            np.linspace(spatial_start - 0.15, spatial_end + 0.15, 50),
            0, 115, alpha=0.04, color="#2E7D32", zorder=0)
        ax.fill_between(
            np.linspace(temporal_start - 0.15, temporal_end + 0.15, 50),
            0, 115, alpha=0.04, color="#C62828", zorder=0)

    # Plot each encoder polygon
    for enc in encoders:
        vals = []
        for i, v in enumerate(raw[enc]):
            span = maxes[i] - mins[i]
            if v is not None and span > 0:
                vals.append((v - mins[i]) / span * 100)
            elif v is not None:
                vals.append(100)  # all encoders equal on this axis
            else:
                vals.append(0)
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', label=ENCODER_LABELS.get(enc, enc).replace('\n', ' '),
                color=ENCODER_COLORS.get(enc, "#888"), linewidth=2, markersize=5)
        ax.fill(angles, vals, color=ENCODER_COLORS.get(enc, "#888"), alpha=0.08)

    # Axis labels with color coding
    ax.set_xticks(angles[:-1])
    label_colors = (["#2E7D32"] * n_spatial +
                    ["#C62828"] * (n_metrics - n_spatial))
    labels = ax.set_xticklabels(all_labels, fontsize=10)
    for label, color in zip(labels, label_colors):
        label.set_color(color)
        label.set_fontweight("bold")

    ax.set_ylim(0, 115)
    # Read clip count from first encoder's metrics (num_clips field)
    first_enc = encoders[0]
    n_clips = all_metrics[first_enc].get("num_clips", "")
    clip_str = f", {n_clips:,} clips" if n_clips else ""
    title = "Spatial + Temporal Encoder Comparison" if has_temporal else "Encoder Comparison"
    ax.set_title(f"{title} (Easy, min-max normalized{clip_str})", fontsize=12,
                 fontweight="bold", pad=25)
    ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.3, -0.05))

    # Add hemisphere labels
    if has_temporal:
        fig.text(0.15, 0.92, "SPATIAL", fontsize=11, fontweight="bold",
                 color="#2E7D32", ha="center")
        fig.text(0.85, 0.92, "TEMPORAL", fontsize=11, fontweight="bold",
                 color="#C62828", ha="center")

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_radar{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_radar.png'}")


# ── Spatial-Temporal Bar Chart (2-row layout) ──────────────────────

def create_grouped_bar_chart(all_metrics: dict, all_temporal: dict, output_dir: Path):
    """2-row bar chart: spatial metrics (top row) | temporal metrics (bottom row).

    Each subplot = one metric, bars = encoders. Green header for spatial,
    red header for temporal. 95% CI error bars on all metrics.
    """
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2:
        return

    spatial_keys = [("prec_at_k", "Prec@K (%)"), ("map_at_k", "mAP@K"),
                    ("cycle_at_k", "Cycle@K (%)"), ("dim_consistency_at_k", "DimConsist@K (%)"),
                    ("ndcg_at_k", "nDCG@K")]
    temporal_keys = [(k, l) for k, l in TEMPORAL_METRICS_DISPLAY if any(
        all_temporal.get(e, {}).get(k) is not None for e in encoders)]

    if any(all_temporal.get(e, {}).get("temporal_locality", {}).get("ratio") is not None
           for e in encoders):
        temporal_keys.append(("temporal_locality_ratio", "Locality Ratio"))

    n_spatial = len(spatial_keys)
    n_temporal = len(temporal_keys)
    n_cols = max(n_spatial, n_temporal) if n_temporal > 0 else n_spatial
    n_rows = 2 if n_temporal > 0 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    def _plot_bar(ax, metric_key, metric_label, source, title_color):
        x = np.arange(len(encoders))
        vals, errs = [], []
        for enc in encoders:
            if source == "spatial":
                v = all_metrics[enc].get("easy", {}).get(metric_key)
                ci_h = all_metrics[enc].get("easy", {}).get("ci", {}).get(
                    metric_key, {}).get("ci_half", 0)
            elif metric_key == "temporal_locality_ratio":
                v = all_temporal.get(enc, {}).get("temporal_locality", {}).get("ratio")
                ci_h = all_temporal.get(enc, {}).get("temporal_locality", {}).get(
                    "ratio_ci", {}).get("ci_half", 0)
            else:
                v = all_temporal.get(enc, {}).get(metric_key)
                ci_h = all_temporal.get(enc, {}).get(f"{metric_key}_ci", {}).get("ci_half", 0)
                if ci_h == 0 and metric_key == "spearman_rho":
                    ci_h = all_temporal.get(enc, {}).get("spearman_rho_ci", {}).get("ci_half", 0)
            vals.append(v if v is not None else 0)
            errs.append(ci_h)

        colors = [ENCODER_COLORS.get(e, "#888") for e in encoders]
        ax.bar(x, vals, color=colors, alpha=0.85, yerr=errs, capsize=4,
               error_kw={"lw": 1.2, "ecolor": "#222"})
        ax.set_title(metric_label, fontsize=10, fontweight="bold", color=title_color)
        ax.set_xticks(x)
        ax.set_xticklabels([e.replace("_", "\n") for e in encoders], fontsize=7)
        ax.tick_params(axis="y", labelsize=8)

        # #78 auto-scale Y: tight around data ± CI so noise-level deltas + error caps
        # are visible. Floor at 0 (percentages); 15 % padding so labels don't clip.
        vals_arr = np.array(vals, dtype=float)
        errs_arr = np.array(errs, dtype=float)
        lo = max(0.0, float((vals_arr - errs_arr).min()))
        hi = float((vals_arr + errs_arr).max())
        pad = max(0.15 * (hi - lo), 0.02 * hi) if hi > 0 else 1
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + e + pad * 0.1, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7, color="#222")

    # Top row: spatial metrics
    for i, (key, label) in enumerate(spatial_keys):
        _plot_bar(axes[0][i], key, label, "spatial", "#2E7D32")
    # Hide unused top-row axes
    for i in range(n_spatial, n_cols):
        axes[0][i].set_visible(False)

    # Bottom row: temporal metrics
    if n_temporal > 0:
        for i, (key, label) in enumerate(temporal_keys):
            _plot_bar(axes[1][i], key, label, "temporal", "#C62828")
        for i in range(n_temporal, n_cols):
            axes[1][i].set_visible(False)

    # Row labels
    fig.text(0.02, 0.75, "SPATIAL", fontsize=13, fontweight="bold",
             color="#2E7D32", rotation=90, va="center")
    if n_temporal > 0:
        fig.text(0.02, 0.28, "TEMPORAL", fontsize=13, fontweight="bold",
                 color="#C62828", rotation=90, va="center")

    plt.suptitle("Spatial vs Temporal Encoder Comparison (Easy mode, 95% CI)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0.04, 0, 1, 0.98])
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_spatial_temporal_bar{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_spatial_temporal_bar.png'}")


# ── Spatial-Temporal Tradeoff Scatter ────────────────────────────────

def create_tradeoff_scatter(all_metrics: dict, all_temporal: dict, output_dir: Path):
    """2D scatter: X=spatial aggregate, Y=temporal aggregate. Each encoder = labeled point."""
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2 or not all_temporal:
        return

    # Spatial aggregate: mean of normalized (Prec@K, mAP@K, nDCG@K)
    spatial_keys = ["prec_at_k", "map_at_k", "ndcg_at_k"]
    # Temporal aggregate: mean of available temporal metrics (normalized)
    temporal_keys = ["spearman_rho", "temporal_prec_at_k", "motion_retrieval_map"]

    # Collect raw values
    spatial_raw = {enc: [] for enc in encoders}
    temporal_raw = {enc: [] for enc in encoders}

    for enc in encoders:
        for sk in spatial_keys:
            v = all_metrics[enc].get("easy", {}).get(sk)
            spatial_raw[enc].append(v if v is not None else 0)
        for tk in temporal_keys:
            v = all_temporal.get(enc, {}).get(tk)
            temporal_raw[enc].append(v if v is not None else 0)

    # Normalize each metric to [0, 1] across encoders
    def normalize_list(raw_dict, keys):
        scores = {}
        for i, k in enumerate(keys):
            vals = [raw_dict[e][i] for e in encoders]
            vmax = max(vals) if max(vals) > 0 else 1
            for e in encoders:
                scores.setdefault(e, []).append(raw_dict[e][i] / vmax)
        return {e: np.mean(v) * 100 for e, v in scores.items()}

    spatial_scores = normalize_list(spatial_raw, spatial_keys)
    temporal_scores = normalize_list(temporal_raw, temporal_keys)

    # Check if temporal has any non-zero values
    if all(v == 0 for v in temporal_scores.values()):
        print("  FATAL: All temporal scores are 0. Run m04d + m06b first.")
        sys.exit(1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for enc in encoders:
        sx = spatial_scores[enc]
        ty = temporal_scores[enc]
        color = ENCODER_COLORS.get(enc, "#888")
        ax.scatter(sx, ty, s=200, c=color, edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(enc.replace("_", "\n"), (sx, ty), textcoords="offset points",
                    xytext=(10, 10), fontsize=9, fontweight="bold", color=color)

    # Diagonal reference
    lim = max(max(spatial_scores.values()), max(temporal_scores.values())) * 1.15
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.3, label="Balanced")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Spatial Score (normalized)", fontsize=11)
    ax.set_ylabel("Temporal Score (normalized)", fontsize=11)
    ax.set_title("Spatial-Temporal Tradeoff", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_tradeoff_scatter{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_tradeoff_scatter.png'}")


# ── Temporal Order Ablation (V-JEPA vs Shuffled) ───────────────────

def create_ablation_chart(all_metrics: dict, all_temporal: dict, output_dir: Path):
    """Paired bar chart: V-JEPA (normal) vs Shuffled on ALL metrics.

    Spatial metrics: shuffled >= normal proves temporal encoding hurts spatial.
    Temporal metrics: normal >> shuffled proves V-JEPA learns temporal structure.
    Delta annotation with propagated 95% CI on each pair.
    """
    if "vjepa" not in all_metrics or "vjepa_shuffled" not in all_metrics:
        print("  SKIP ablation chart: need both vjepa and vjepa_shuffled")
        return

    # Collect all metrics for both encoders
    spatial_defs = [
        ("prec_at_k", "Prec@K (%)", "spatial"),
        ("map_at_k", "mAP@K", "spatial"),
        ("cycle_at_k", "Cycle@K (%)", "spatial"),
        ("dim_consistency_at_k", "DimConsist@K (%)", "spatial"),
        ("ndcg_at_k", "nDCG@K", "spatial"),
    ]
    temporal_defs = [
        ("spearman_rho", "Spearman \u03c1", "temporal"),
        ("temporal_prec_at_k", "Temp Prec@K (%)", "temporal"),
        ("motion_retrieval_map", "Motion mAP", "temporal"),
    ]
    # Add temporal locality if available
    if all_temporal and all_temporal.get("vjepa", {}).get("temporal_locality", {}).get("ratio") is not None:
        temporal_defs.append(("temporal_locality_ratio", "Locality\n(lower=better)", "temporal"))

    has_temporal = all_temporal and any(
        all_temporal.get("vjepa", {}).get(k) is not None for k, _, _ in temporal_defs[:3])

    n_spatial_d = len(spatial_defs)
    n_temporal_d = len(temporal_defs) if has_temporal else 0
    n_cols = max(n_spatial_d, n_temporal_d) if n_temporal_d > 0 else n_spatial_d
    n_rows = 2 if n_temporal_d > 0 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    def _ablation_bar(ax, key, label, domain):
        vals, errs = [], []
        for enc in ["vjepa", "vjepa_shuffled"]:
            if domain == "spatial":
                v = all_metrics[enc].get("easy", {}).get(key)
                ci_h = all_metrics[enc].get("easy", {}).get("ci", {}).get(
                    key, {}).get("ci_half", 0)
            elif key == "temporal_locality_ratio":
                v = (all_temporal or {}).get(enc, {}).get(
                    "temporal_locality", {}).get("ratio")
                ci_h = 0
            else:
                v = (all_temporal or {}).get(enc, {}).get(key)
                ci_h = (all_temporal or {}).get(enc, {}).get(
                    f"{key}_ci", {}).get("ci_half", 0)
                if ci_h == 0 and key == "spearman_rho":
                    ci_h = (all_temporal or {}).get(enc, {}).get(
                        "spearman_rho_ci", {}).get("ci_half", 0)
            vals.append(v if v is not None else 0)
            errs.append(ci_h)

        normal_v, shuffled_v = vals
        normal_e, shuffled_e = errs
        x = np.array([0, 1])
        ax.bar(x, vals, color=["#2196F3", "#E91E63"], alpha=0.85,
               yerr=errs, capsize=5, error_kw={"lw": 1.5})

        delta = normal_v - shuffled_v
        delta_ci = np.sqrt(normal_e**2 + shuffled_e**2)
        delta_sign = "+" if delta > 0 else ""
        if key == "temporal_locality_ratio":
            expected = delta < 0
        elif domain == "temporal":
            expected = delta > 0
        else:
            expected = delta < 0 or key == "cycle_at_k"
        delta_color = "#2E7D32" if expected else "#C62828"
        ci_str = f" \u00b1{delta_ci:.2f}" if delta_ci > 0 else ""
        ax.annotate(f"\u0394={delta_sign}{delta:.2f}{ci_str}",
                    xy=(0.5, max(vals) * 1.05), fontsize=7, fontweight="bold",
                    color=delta_color, ha="center")
        title_color = "#2E7D32" if domain == "spatial" else "#C62828"
        ax.set_title(label, fontsize=9, fontweight="bold", color=title_color)
        ax.set_xticks(x)
        ax.set_xticklabels(["V-JEPA\n(normal)", "V-JEPA\n(shuffled)"], fontsize=7)
        ax.tick_params(axis="y", labelsize=8)

        # #81 tight Y auto-scale — matches create_bar_chart / create_grouped_bar_chart.
        # Default matplotlib zoom anchored at 0 wasted space on high-value metrics
        # (Cycle@K 70+, DimConsist@K 48+). Now: [max(0, min-err), max+err] + 15 % pad.
        vals_arr = np.array(vals, dtype=float)
        errs_arr = np.array(errs, dtype=float)
        lo = max(0.0, float((vals_arr - errs_arr).min()))
        hi = float((vals_arr + errs_arr).max())
        pad = max(0.15 * (hi - lo), 0.02 * hi) if hi > 0 else 1
        ax.set_ylim(max(0.0, lo - pad), hi + pad)

    # Top row: spatial
    for i, (key, label, domain) in enumerate(spatial_defs):
        _ablation_bar(axes[0][i], key, label, domain)
    for i in range(n_spatial_d, n_cols):
        axes[0][i].set_visible(False)

    # Bottom row: temporal
    if n_temporal_d > 0:
        for i, (key, label, domain) in enumerate(temporal_defs):
            _ablation_bar(axes[1][i], key, label, domain)
        for i in range(n_temporal_d, n_cols):
            axes[1][i].set_visible(False)

    fig.text(0.02, 0.75, "SPATIAL", fontsize=13, fontweight="bold",
             color="#2E7D32", rotation=90, va="center")
    if n_temporal_d > 0:
        fig.text(0.02, 0.28, "TEMPORAL", fontsize=13, fontweight="bold",
                 color="#C62828", rotation=90, va="center")

    plt.suptitle("Temporal Order Ablation: Does Frame Order Matter?",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, -0.01,
             "Green \u0394 = expected  |  "
             "Red \u0394 = unexpected (temporal encoding not helping)",
             ha="center", fontsize=8, color="#555", style="italic")
    plt.tight_layout(rect=[0.04, 0.02, 1, 0.98])
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_temporal_ablation{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_temporal_ablation.png'}")


# ── Adaptation Stage Ablation (frozen vs pretrained vs surgical) ──

def create_adaptation_ablation(all_metrics: dict, all_temporal: dict, output_dir: Path):
    """Adaptation stage comparison: frozen vs pretrained vs surgical.

    Auto-detects adapted encoders (vjepa_lambda*, vjepa_surgical*).
    Delta + propagated 95% CI vs frozen baseline. If CI doesn't cross
    zero, the improvement is statistically significant.
    """

    baseline = "vjepa"
    if baseline not in all_metrics:
        print("  SKIP adaptation ablation: need frozen vjepa as baseline")
        return

    # Auto-detect adapted and surgical encoders
    stages = [(baseline, "V-JEPA 2\n(frozen)", "#2196F3")]
    _adapted_colors = ["#7B1FA2", "#9C27B0", "#CE93D8"]
    _surgical_colors = ["#00695C", "#00897B", "#4DB6AC"]
    adapted_idx, surgical_idx = 0, 0

    for enc in sorted(all_metrics.keys()):
        if enc == baseline:
            continue
        m_lam = re.match(r'vjepa_lambda(\d+(?:_\d+)*)', enc)
        if m_lam and "shuffled" not in enc:
            lam_str = m_lam.group(1).replace("_", ".", 1)
            stages.append((enc, f"Pretrained\n(\u03bb={lam_str})",
                           _adapted_colors[adapted_idx % len(_adapted_colors)]))
            adapted_idx += 1
        elif enc == "vjepa_adapted":
            stages.append((enc, "Pretrained", _adapted_colors[adapted_idx % len(_adapted_colors)]))
            adapted_idx += 1
        elif enc.startswith("vjepa_surgical"):
            stages.append((enc, f"Surgical\n({enc.split('_', 2)[-1]})",
                           _surgical_colors[surgical_idx % len(_surgical_colors)]))
            surgical_idx += 1

    if len(stages) < 2:
        print("  SKIP adaptation ablation: need at least 1 adapted/surgical encoder")
        return

    stage_labels = [s[1] for s in stages]
    stage_colors = [s[2] for s in stages]
    n_stages = len(stages)

    # Metric definitions
    spatial_defs = [
        ("prec_at_k", "Prec@K (%)", "spatial"),
        ("map_at_k", "mAP@K", "spatial"),
        ("cycle_at_k", "Cycle@K (%)", "spatial"),
        ("dim_consistency_at_k", "DimConsist@K (%)", "spatial"),
        ("ndcg_at_k", "nDCG@K", "spatial"),
    ]
    temporal_defs = [
        ("spearman_rho", "Spearman \u03c1", "temporal"),
        ("temporal_prec_at_k", "Temp Prec@K (%)", "temporal"),
        ("motion_retrieval_map", "Motion mAP", "temporal"),
    ]
    if all_temporal and any(
            all_temporal.get(baseline, {}).get("temporal_locality", {}).get("ratio") is not None
            for _ in [1]):
        temporal_defs.append(("temporal_locality_ratio", "Locality\n(lower=better)", "temporal"))

    has_temporal = all_temporal and any(
        all_temporal.get(baseline, {}).get(k) is not None for k, _, _ in temporal_defs[:3])

    n_spatial_d = len(spatial_defs)
    n_temporal_d = len(temporal_defs) if has_temporal else 0
    n_cols = max(n_spatial_d, n_temporal_d) if n_temporal_d > 0 else n_spatial_d
    n_rows = 2 if n_temporal_d > 0 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    def _get_val_ci(enc, key, domain):
        if domain == "spatial":
            v = all_metrics[enc].get("easy", {}).get(key)
            ci_h = all_metrics[enc].get("easy", {}).get("ci", {}).get(
                key, {}).get("ci_half", 0)
        elif key == "temporal_locality_ratio":
            v = (all_temporal or {}).get(enc, {}).get(
                "temporal_locality", {}).get("ratio")
            ci_h = (all_temporal or {}).get(enc, {}).get(
                "temporal_locality", {}).get("ratio_ci", {}).get("ci_half", 0)
        else:
            v = (all_temporal or {}).get(enc, {}).get(key)
            ci_h = (all_temporal or {}).get(enc, {}).get(
                f"{key}_ci", {}).get("ci_half", 0)
            if ci_h == 0 and key == "spearman_rho":
                ci_h = (all_temporal or {}).get(enc, {}).get(
                    "spearman_rho_ci", {}).get("ci_half", 0)
        return (v if v is not None else 0), ci_h

    def _adaptation_bar(ax, key, label, domain):
        vals, errs = [], []
        for enc, _, _ in stages:
            v, ci_h = _get_val_ci(enc, key, domain)
            vals.append(v)
            errs.append(ci_h)

        x = np.arange(n_stages)
        bar_w = 0.6
        ax.bar(x, vals, bar_w, color=stage_colors, alpha=0.85,
               yerr=errs, capsize=4, error_kw={"lw": 1.5})

        # Delta annotations: each adapted stage vs frozen baseline
        baseline_v, baseline_e = vals[0], errs[0]
        for si in range(1, n_stages):
            delta = vals[si] - baseline_v
            delta_ci = np.sqrt(errs[si]**2 + baseline_e**2)
            delta_sign = "+" if delta > 0 else ""
            # Significance: CI doesn't cross zero
            significant = abs(delta) > delta_ci and delta_ci > 0
            if key == "temporal_locality_ratio":
                improved = delta < 0  # lower is better
            else:
                improved = delta > 0
            if significant:
                delta_color = "#2E7D32" if improved else "#C62828"
                sig_marker = " *"
            else:
                delta_color = "#888888"
                sig_marker = ""
            ci_str = f"\u00b1{delta_ci:.2f}" if delta_ci > 0 else ""
            annot = f"\u0394={delta_sign}{delta:.2f} {ci_str}{sig_marker}"
            y_pos = max(vals[si] + errs[si], baseline_v + baseline_e) * 1.02
            ax.annotate(annot, xy=(x[si], y_pos), fontsize=6, fontweight="bold",
                        color=delta_color, ha="center", va="bottom")

        title_color = "#2E7D32" if domain == "spatial" else "#C62828"
        ax.set_title(label, fontsize=9, fontweight="bold", color=title_color)
        ax.set_xticks(x)
        ax.set_xticklabels([sl.replace("\n", " ") for sl in stage_labels],
                           fontsize=6, rotation=15, ha="right")
        ax.tick_params(axis="y", labelsize=8)

        # #81 tight Y auto-scale — matches create_bar_chart pattern. Default
        # zero-anchored zoom wasted vertical space on high-value metrics.
        vals_arr = np.array(vals, dtype=float)
        errs_arr = np.array(errs, dtype=float)
        lo = max(0.0, float((vals_arr - errs_arr).min()))
        hi = float((vals_arr + errs_arr).max())
        pad = max(0.15 * (hi - lo), 0.02 * hi) if hi > 0 else 1
        # Include delta-annotation head-room (y_pos uses max(vals+errs)*1.02 above)
        ax.set_ylim(max(0.0, lo - pad), hi + pad * 1.3)

    # Top row: spatial
    for i, (key, label, domain) in enumerate(spatial_defs):
        _adaptation_bar(axes[0][i], key, label, domain)
    for i in range(n_spatial_d, n_cols):
        axes[0][i].set_visible(False)

    # Bottom row: temporal
    if n_temporal_d > 0:
        for i, (key, label, domain) in enumerate(temporal_defs):
            _adaptation_bar(axes[1][i], key, label, domain)
        for i in range(n_temporal_d, n_cols):
            axes[1][i].set_visible(False)

    fig.text(0.02, 0.75, "SPATIAL", fontsize=13, fontweight="bold",
             color="#2E7D32", rotation=90, va="center")
    if n_temporal_d > 0:
        fig.text(0.02, 0.28, "TEMPORAL", fontsize=13, fontweight="bold",
                 color="#C62828", rotation=90, va="center")

    plt.suptitle("Adaptation Stage Ablation: Frozen vs Pretrained vs Surgical",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, -0.01,
             "Green \u0394* = significant improvement (CI excludes 0)  |  "
             "Red \u0394* = significant degradation  |  "
             "Gray \u0394 = not significant",
             ha="center", fontsize=7, color="#555", style="italic")
    plt.tight_layout(rect=[0.04, 0.02, 1, 0.98])
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_adaptation_ablation{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_adaptation_ablation.png'}")


# ── Normalized Heatmap (Encoders × Metrics) ────────────────────────

def create_heatmap(all_metrics: dict, all_temporal: dict, output_dir: Path):
    """Encoder × metric heatmap. Spatial columns left, temporal right.

    Cell text shows value ± CI. Color intensity = normalized rank across
    encoders (per column). Visual: DINOv2 row hot left, V-JEPA hot right.
    """
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2:
        return

    spatial_defs = [
        ("prec_at_k", "Prec@K"),
        ("map_at_k", "mAP@K"),
        ("cycle_at_k", "Cycle@K"),
        ("dim_consistency_at_k", "DimConsist@K"),
        ("ndcg_at_k", "nDCG@K"),
    ]
    temporal_defs = []
    if all_temporal and any(
            all_temporal.get(e, {}).get("spearman_rho") is not None for e in encoders):
        temporal_defs = [
            ("spearman_rho", "Spearman \u03c1"),
            ("temporal_prec_at_k", "Temp Prec@K"),
            ("motion_retrieval_map", "Motion mAP"),
        ]

    all_defs = spatial_defs + temporal_defs
    n_spatial = len(spatial_defs)
    n_metrics = len(all_defs)
    n_enc = len(encoders)

    # Build value + CI matrices
    val_matrix = np.zeros((n_enc, n_metrics))
    ci_matrix = np.zeros((n_enc, n_metrics))

    for ei, enc in enumerate(encoders):
        for mi, (key, _) in enumerate(all_defs):
            if mi < n_spatial:
                v = all_metrics[enc].get("easy", {}).get(key)
                ci_h = all_metrics[enc].get("easy", {}).get("ci", {}).get(
                    key, {}).get("ci_half", 0)
            else:
                v = (all_temporal or {}).get(enc, {}).get(key)
                ci_h = (all_temporal or {}).get(enc, {}).get(
                    f"{key}_ci", {}).get("ci_half", 0)
                if ci_h == 0 and key == "spearman_rho":
                    ci_h = (all_temporal or {}).get(enc, {}).get(
                        "spearman_rho_ci", {}).get("ci_half", 0)
            val_matrix[ei, mi] = v if v is not None else 0
            # Scale CI to match display units: cycle_at_k CI is in proportion
            # (0.008) but value is percentage (76.0) — multiply CI by 100
            if key == "cycle_at_k" and ci_h > 0 and ci_h < 1:
                ci_h = ci_h * 100
            ci_matrix[ei, mi] = ci_h

    # Normalize per column to [0, 1] for color mapping
    norm_matrix = np.zeros_like(val_matrix)
    for mi in range(n_metrics):
        col = val_matrix[:, mi]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            norm_matrix[:, mi] = (col - cmin) / (cmax - cmin)
        else:
            norm_matrix[:, mi] = 0.5

    # Custom colormap: spatial=green shades, temporal=red shades
    from matplotlib.colors import LinearSegmentedColormap
    spatial_cmap = LinearSegmentedColormap.from_list("spatial", ["#FFFFFF", "#2E7D32"])
    temporal_cmap = LinearSegmentedColormap.from_list("temporal", ["#FFFFFF", "#C62828"])

    fig, ax = plt.subplots(1, 1, figsize=(max(10, 1.8 * n_metrics), 0.9 * n_enc + 2))

    # Draw cells with appropriate colormap per column
    for mi in range(n_metrics):
        cmap = spatial_cmap if mi < n_spatial else temporal_cmap
        for ei in range(n_enc):
            color = cmap(norm_matrix[ei, mi] * 0.85 + 0.05)  # avoid pure white
            ax.add_patch(plt.Rectangle((mi, n_enc - 1 - ei), 1, 1,
                                       facecolor=color, edgecolor="white", lw=2))
            # Cell text: value ± CI
            v = val_matrix[ei, mi]
            ci = ci_matrix[ei, mi]
            key = all_defs[mi][0]
            if key in ("prec_at_k", "cycle_at_k", "dim_consistency_at_k", "temporal_prec_at_k"):
                txt = f"{v:.1f}"
                if ci > 0:
                    txt += f"\n\u00b1{ci:.1f}"
            else:
                txt = f"{v:.3f}"
                if ci > 0:
                    txt += f"\n\u00b1{ci:.3f}"
            text_color = "white" if norm_matrix[ei, mi] > 0.6 else "black"
            ax.text(mi + 0.5, n_enc - 0.5 - ei, txt, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=text_color)

    # Separator line between spatial and temporal
    if temporal_defs:
        ax.axvline(x=n_spatial, color="#333", linewidth=2, linestyle="-")

    # Labels
    enc_labels = [ENCODER_LABELS.get(e, e).replace('\n', ' ') for e in encoders]
    ax.set_yticks(np.arange(n_enc) + 0.5)
    ax.set_yticklabels(enc_labels[::-1], fontsize=9)
    ax.set_xticks(np.arange(n_metrics) + 0.5)
    metric_labels = [d[1] for d in all_defs]
    xlabels = ax.set_xticklabels(metric_labels, fontsize=9, rotation=30, ha="right")
    for i, xl in enumerate(xlabels):
        xl.set_color("#2E7D32" if i < n_spatial else "#C62828")
        xl.set_fontweight("bold")

    ax.set_xlim(0, n_metrics)
    ax.set_ylim(0, n_enc)

    # Group headers
    if temporal_defs:
        ax.text(n_spatial / 2, n_enc + 0.3, "SPATIAL", ha="center",
                fontsize=12, fontweight="bold", color="#2E7D32")
        ax.text(n_spatial + len(temporal_defs) / 2, n_enc + 0.3, "TEMPORAL",
                ha="center", fontsize=12, fontweight="bold", color="#C62828")

    ax.set_title("Encoder × Metric Performance (normalized per column, 95% CI)",
                 fontsize=12, fontweight="bold", pad=25)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_heatmap{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_heatmap.png'}")


# ── LaTeX Table ──────────────────────────────────────────────────────

def create_latex_table(all_metrics: dict, output_dir: Path):
    """Generate LaTeX table for paper (Easy mode)."""
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if not encoders:
        return

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Encoder comparison on 10K Indian urban clips (Easy mode, $k=6$)}")
    lines.append(r"\label{tab:encoder_comparison}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Encoder & Dim & Cycle@K & Overlap@K & Prec@K & mAP@K & nDCG@K \\")
    lines.append(r"\midrule")

    for enc in encoders:
        easy = all_metrics[enc].get("easy", {})
        dim = all_metrics[enc].get("encoder_dim", get_encoder_info(enc)["dim"])
        label = enc.replace("_", r"\_")

        ci_data = easy.get("ci", {})
        vals = []
        for key in ["cycle_at_k", "dim_consistency_at_k", "prec_at_k", "map_at_k", "ndcg_at_k"]:
            v = easy.get(key)
            ci_half = ci_data.get(key, {}).get("ci_half")
            if v is None:
                vals.append("--")
            elif key in ("cycle_at_k", "dim_consistency_at_k", "prec_at_k"):
                if ci_half is not None:
                    vals.append(f"{v:.1f}{{\\tiny$\\pm${ci_half:.1f}}}")
                else:
                    vals.append(f"{v:.2f}")
            else:
                if ci_half is not None:
                    vals.append(f"{v:.3f}{{\\tiny$\\pm${ci_half:.3f}}}")
                else:
                    vals.append(f"{v:.4f}")

        lines.append(f"{label} & {dim} & {' & '.join(vals)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_file = output_dir / "m08b_comparison_table.tex"
    with open(tex_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {tex_file}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-encoder comparison plots + LaTeX table (CPU-only)")
    parser.add_argument("--SANITY", action="store_true", help="Placeholder for consistency")
    parser.add_argument("--POC", action="store_true", help="POC subset (~10K clips)")
    parser.add_argument("--FULL", action="store_true", help="Process all available encoders")
    parser.add_argument("--encoders", type=str, default=None,
                        help="Comma-separated encoder names to compare "
                             "(e.g., vjepa,vjepa_adapted,vjepa_lambda0_01). "
                             "If omitted, uses all registered encoders.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Per-variant output dir override. When omitted, "
                             "writes to outputs/full/m08b_compare/ (shared — "
                             "collides if multiple variants run eval in parallel). "
                             "run_eval.sh passes <yaml.data.output_dir>/eval/ "
                             "so paired_bootstrap_results.json + plots stay per-variant.")
    add_subset_arg(parser)
    add_wandb_args(parser)
    # NO --cache-policy arg here on purpose — see module docstring DELETE-PROTECTION
    # POLICY block (m08b is a pure visualization function, always recomputes).
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    # Always wipe + recreate the output_dir at startup — m08b owns it exclusively.
    # When called from run_eval.sh, args.output_dir = "${OUT_DIR}/eval" (per-variant,
    # single-owner) — this rmtree only nukes that subdir, NOT the m09 training parent.
    # When called standalone, it falls back to get_module_output_dir which is also
    # single-owner (m08b_compare/). Stale-plot bug fix — see module docstring reason #3.
    if args.output_dir:
        _m08b_out = Path(args.output_dir)
    else:
        _m08b_out = get_module_output_dir(
            "m08b_compare", args.subset, sanity=args.SANITY, poc=args.POC)
    if _m08b_out.exists():
        n_files = sum(1 for _ in _m08b_out.rglob("*") if _.is_file())
        print(f"  [m08b] wiping output_dir ({_m08b_out.name}) "
              f"({n_files} stale file(s)) — always-recompute policy, see docstring")
        shutil.rmtree(_m08b_out)
    _m08b_out.mkdir(parents=True, exist_ok=True)

    # Parse custom encoder list
    encoder_list = None
    if args.encoders:
        encoder_list = [e.strip() for e in args.encoders.split(",")]
        # Auto-assign labels/colors for unregistered encoders
        _extra_colors = ["#7B1FA2", "#00695C", "#BF360C", "#1A237E",
                         "#33691E", "#880E4F", "#004D40", "#F57F17"]
        for i, enc in enumerate(encoder_list):
            if enc not in ENCODER_LABELS:
                # Dynamic labels for lambda encoders and their shuffled variants
                m_shuf = re.match(r'vjepa_lambda(\d+(?:_\d+)*)_shuffled', enc)
                m_lam = re.match(r'vjepa_lambda(\d+(?:_\d+)*)', enc)
                if m_shuf:
                    lam_str = m_shuf.group(1).replace("_", ".", 1)
                    ENCODER_LABELS[enc] = f"Shuffled Adapted\n(\u03bb={lam_str})"
                elif m_lam:
                    lam_str = m_lam.group(1).replace("_", ".", 1)
                    ENCODER_LABELS[enc] = f"V-JEPA Adapted\n(\u03bb={lam_str})"
                else:
                    ENCODER_LABELS[enc] = enc.replace("_", "\n")
                ENCODER_COLORS[enc] = _extra_colors[i % len(_extra_colors)]
        # Override ENCODER_ORDER with custom list so all plot functions include them
        global ENCODER_ORDER
        ENCODER_ORDER = encoder_list

    base_dir = get_output_dir(args.subset, sanity=args.SANITY, poc=args.POC)
    # Per-variant output_dir override prevents 4-way overwrite when multiple
    # variants run eval in parallel (paired_bootstrap_results.json + plots).
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = get_module_output_dir("m08b_compare", args.subset, sanity=args.SANITY, poc=args.POC)
    # Per-module restructure (#64): m06 JSONs live under <base_dir>/m06_faiss_metrics/,
    # not flat at <base_dir>/. Analogous fix at m06_faiss_metrics.py — m08b was missed.
    # errors_N_fixes #75. metrics_dir stays SHARED (all variants read each other's m06 metrics).
    metrics_dir = base_dir / "m06_faiss_metrics"
    print(f"Output dir: {output_dir}")
    print(f"Scanning for encoder metrics in: {metrics_dir}")

    all_metrics = load_all_metrics(metrics_dir, encoder_list=encoder_list)
    if not all_metrics:
        print(f"FATAL: No encoder metrics found in {metrics_dir}. Run m06 first.")
        sys.exit(1)

    all_temporal = load_all_temporal(metrics_dir, encoder_list=encoder_list)

    print(f"\nFound {len(all_metrics)} encoder(s): {', '.join(all_metrics.keys())}")
    if all_temporal:
        print(f"Found {len(all_temporal)} temporal result(s): {', '.join(all_temporal.keys())}")

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb_run = init_wandb("m08b", mode, config=vars(args), enabled=not args.no_wandb)

    # Terminal table
    print_summary_table(all_metrics)

    # Plots + table (need >= 2 encoders for comparison)
    if len(all_metrics) >= 2:
        pbar = make_pbar(total=8, desc="m08b_compare", unit="plot")
        create_bar_chart(all_metrics, output_dir)
        pbar.update(1)
        # #77: radar requires ≥3 encoders to be informative. With only 2, per-axis
        # min-max normalization forces one to 100 and the other to 0 (= center) on
        # every axis — the visualization becomes winner-takes-all binary and
        # misleads readers into inferring a lopsided result when deltas are at
        # noise level. Skipped until ExPLoRA arm lands post-v11 (runbook Step G).
        if len(all_metrics) >= 3:
            create_radar_plot(all_metrics, output_dir, all_temporal=all_temporal)
        else:
            print(f"  [m08b] skipping radar: n_encoders={len(all_metrics)} < 3 — "
                  f"min-max normalization degenerates to binary winner/loser "
                  f"with 2 encoders. Re-runs automatically when ExPLoRA arm lands.")
            # Unconditional stale-radar purge — m08b is always-recompute (see docstring).
            # output_dir was wiped at startup in main(), so this is belt-and-suspenders
            # for cases where create_radar_plot wrote a partial file before erroring.
            for ext in (".png", ".pdf"):
                stale = output_dir / f"m08b_radar{ext}"
                if stale.exists():
                    stale.unlink()
                    print(f"  [m08b] purged stale radar {stale.name}")
        pbar.update(1)
        create_latex_table(all_metrics, output_dir)
        pbar.update(1)
        create_grouped_bar_chart(all_metrics, all_temporal, output_dir)
        pbar.update(1)
        create_tradeoff_scatter(all_metrics, all_temporal, output_dir)
        pbar.update(1)
        create_ablation_chart(all_metrics, all_temporal, output_dir)
        pbar.update(1)
        create_adaptation_ablation(all_metrics, all_temporal, output_dir)
        pbar.update(1)
        create_heatmap(all_metrics, all_temporal, output_dir)
        pbar.update(1)
        pbar.close()

        for name in ["m08b_encoder_comparison", "m08b_radar",
                     "m08b_spatial_temporal_bar", "m08b_tradeoff_scatter",
                     "m08b_temporal_ablation", "m08b_adaptation_ablation",
                     "m08b_heatmap"]:
            png = output_dir / f"{name}.png"
            if png.exists():
                log_image(wb_run, name, str(png))
    else:
        print("Only 1 encoder found — skipping comparison plots (need >= 2).")

    # iter10 Option C: paired bootstrap on per-clip deltas.
    # Needs BOTH a frozen arm and ≥1 adapted arm (surgical/explora/adapted) with
    # per_clip_{encoder}_{mode}.npz files written by m06 (iter10 patch). If either
    # arm lacks the npz, skip cleanly.
    compute_paired_bootstrap(metrics_dir, output_dir, encoder_list=encoder_list)

    # Δ-vs-frozen chart with PAIRED CI (errors_N_fixes #80 follow-up). The 8 main
    # plots above show INDEPENDENT CIs (~±3.5 pp at N=308) which visually overlap
    # and hide significance — this chart uses paired CIs (~±0.6 pp) so 0.87 pp
    # gains become visually obvious. No-op if paired_bootstrap_results.json missing.
    create_paired_delta_chart(output_dir)
    paired_png = output_dir / "m08b_paired_delta.png"
    if paired_png.exists():
        log_image(wb_run, "m08b_paired_delta", str(paired_png))

    finish_wandb(wb_run)
    print("\n=== COMPARISON COMPLETE ===")


if __name__ == "__main__":
    main()
