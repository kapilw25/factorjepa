"""Regenerate all m09c plots from disk (probe_history.jsonl + training_summary.json).

Standalone regen tool — applies the latest plot-code patches (#79/#80/#81) without
waiting for m09c's next probe (Python load-once prevents mid-run code pickup).

    python -u scripts/regen_m09c_plots.py                 # defaults to outputs/full/m09c_surgery
    python -u scripts/regen_m09c_plots.py --mode full
    python -u scripts/regen_m09c_plots.py --dir outputs/poc/m09c_surgery
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


# ── Config helpers ───────────────────────────────────────────────────

def load_thresholds(yaml_path: Path, mode: str = "full") -> dict:
    """Read early-stop thresholds from ch11_surgery.yaml (same keys m09c uses at train time)."""
    cfg = yaml.safe_load(open(yaml_path))
    probe = cfg["probe"]
    return {
        "forgetting_threshold_pct": probe["forgetting_threshold_pct"],
        "forgetting_patience":      probe["forgetting_patience"],
        "bwt_trigger_enabled":      probe["bwt_trigger_enabled"][mode],
        "bwt_ci_fraction":          probe["bwt_ci_fraction"],
        "bwt_absolute_floor":       probe["bwt_absolute_floor"],
        "bwt_patience":             probe["bwt_patience"],
    }


# ── Plot 1: probe_trajectory.png ────────────────────────────────────

def plot_probe_trajectory(output_dir: Path, probe_history: list):
    """3-panel: Prec@K, mAP@K, Cycle@K — per-panel tight zoom on MEAN only (#81)."""
    steps_ = [r["global_step"] for r in probe_history]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    panels = [
        (axes[0], "prec_at_k",  "Prec@K (%)",  "#2E7D32"),
        (axes[1], "map_at_k",   "mAP@K (%)",   "#1565C0"),
        (axes[2], "cycle_at_k", "Cycle@K (%)", "#E65100"),
    ]
    for ax_p, metric, label, color in panels:
        means  = [r[metric]["mean"] for r in probe_history]
        ci_los = [r[metric]["ci_lo"] for r in probe_history]
        ci_his = [r[metric]["ci_hi"] for r in probe_history]
        ax_p.plot(steps_, means, "o-", color=color, linewidth=2, markersize=6, alpha=0.95)
        ax_p.fill_between(steps_, ci_los, ci_his, color=color, alpha=0.18)
        seen = set()
        for r in probe_history:
            if r["stage_idx"] not in seen:
                ax_p.axvline(r["global_step"], color="gray", linestyle=":",
                             alpha=0.5, linewidth=1)
                seen.add(r["stage_idx"])
        ax_p.annotate(f"end={means[-1]:.2f}",
                      xy=(steps_[-1], means[-1]),
                      xytext=(5, 0), textcoords="offset points",
                      fontsize=8, color=color, va="center")
        ax_p.set_ylabel(label, color=color, fontsize=10, fontweight="bold")
        ax_p.tick_params(axis="y", labelsize=9, colors=color)
        ax_p.grid(True, alpha=0.3)
        # #81 tight zoom on MEAN trajectory — target ≥90 % of panel filled by mean.
        # pad = spread / 18 gives 90 % utilization; floor 0.005 keeps axis visible.
        mean_lo, mean_hi = min(means), max(means)
        pad = max((mean_hi - mean_lo) / 18, 0.005)
        ax_p.set_ylim(mean_lo - pad, mean_hi + pad)
        # Per-point value labels
        for s, m in zip(steps_, means):
            ax_p.annotate(f"{m:.2f}", xy=(s, m), xytext=(0, 8),
                          textcoords="offset points", fontsize=7, color=color, ha="center")

    axes[-1].set_xlabel("Optimizer step", fontsize=10)
    fig.suptitle(f"Probe trajectory (N={probe_history[0]['num_clips']} val-split, "
                 f"{len(probe_history)} probes, 95 % BCa CI) — per-panel y-axis tight on mean (#81)",
                 fontsize=10, fontweight="bold", y=0.995)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(output_dir / f"probe_trajectory{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()


# ── Plot 2: m09_forgetting.png ──────────────────────────────────────

def plot_forgetting(output_dir: Path, probe_history: list, best_state: dict,
                    kill_state: dict, thresholds: dict):
    """Prec@K + running max + tolerance band + BWT twin axis — tight on mean (#80)."""
    steps_ = [r["global_step"] for r in probe_history]
    prec_ = [r["prec_at_k"]["mean"] for r in probe_history]
    running_max = []
    cur_max = -1.0
    for v in prec_:
        cur_max = max(cur_max, v); running_max.append(cur_max)
    strike_x, strike_y = [], []
    for s, p, m in zip(steps_, prec_, running_max):
        if m > 0 and p < m - thresholds["forgetting_threshold_pct"]:
            strike_x.append(s); strike_y.append(p)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps_, prec_, "o-", color="#2E7D32", linewidth=2,
            label="Prec@K (held-out val)", markersize=6, alpha=0.9)
    ax.plot(steps_, running_max, "--", color="#1565C0", linewidth=1.5,
            label="running max", alpha=0.8)
    threshold_low = [m - thresholds["forgetting_threshold_pct"] for m in running_max]
    ax.fill_between(steps_, threshold_low, running_max, color="#FFA726", alpha=0.15,
                    label=f"forgetting tolerance ({thresholds['forgetting_threshold_pct']:.1f} pp)")
    if strike_x:
        ax.scatter(strike_x, strike_y, marker="x", s=100, color="#C62828",
                   label=f"forgetting strikes ({len(strike_x)})", zorder=5, linewidths=2)
    seen = set()
    for r in probe_history:
        if r["stage_idx"] not in seen:
            ax.axvline(r["global_step"], color="gray", linestyle=":",
                       alpha=0.4, linewidth=1)
            seen.add(r["stage_idx"])
    if best_state.get("global_step", -1) >= 0:
        ax.axvline(best_state["global_step"], color="#2E7D32", linestyle="-.",
                   alpha=0.5, linewidth=1.5,
                   label=f"best ckpt (Prec@K={best_state['prec_at_k']:.2f})")

    ax2 = ax.twinx()
    bwt_vals = [r.get("bwt", r["prec_at_k"]["mean"] - prec_[0]) for r in probe_history]
    ax2.plot(steps_, bwt_vals, "s-", color="#8E24AA", linewidth=1.8, markersize=5,
             alpha=0.9, label=f"BWT (latest={bwt_vals[-1]:+.2f} pp)")
    ax2.axhline(0.0, color="#8E24AA", linestyle=":", alpha=0.4, linewidth=1)
    if thresholds["bwt_trigger_enabled"]:
        latest_ci = probe_history[-1]["prec_at_k"]["ci_half"]
        ci_thr_vals = [-(thresholds["bwt_ci_fraction"] * r["prec_at_k"]["ci_half"])
                       for r in probe_history]
        ax2.plot(steps_, ci_thr_vals, color="#C62828", linestyle="--", alpha=0.6, linewidth=1,
                 label=f"ci-threshold (−{thresholds['bwt_ci_fraction']:.2f}×CI_half, "
                       f"now={-thresholds['bwt_ci_fraction']*latest_ci:+.2f})")
        ax2.axhline(-thresholds["bwt_absolute_floor"], color="#E65100",
                    linestyle=":", alpha=0.7, linewidth=1.2,
                    label=f"abs floor (−{thresholds['bwt_absolute_floor']:.2f} pp "
                          f"× {thresholds['bwt_patience']} probes)")
    ax2.set_ylabel("BWT (pp) = Prec@K − Prec@K[first]", color="#8E24AA")
    ax2.tick_params(axis="y", labelcolor="#8E24AA")

    # #80 mean-only tight zoom — target ≥90 % of panel filled by trajectory.
    prec_lo_tight = min(prec_)
    prec_hi_tight = max(max(prec_), max(running_max))
    prec_pad = max((prec_hi_tight - prec_lo_tight) / 18, 0.005)
    ax.set_ylim(prec_lo_tight - prec_pad, prec_hi_tight + prec_pad)
    bwt_lo_tight = min(bwt_vals); bwt_hi_tight = max(bwt_vals)
    bwt_pad = max((bwt_hi_tight - bwt_lo_tight) / 18, 0.005)
    ax2.set_ylim(bwt_lo_tight - bwt_pad, bwt_hi_tight + bwt_pad)

    # Value labels on markers
    for s, p in zip(steps_, prec_):
        ax.annotate(f"{p:.3f}", xy=(s, p), xytext=(5, 8), textcoords="offset points",
                    fontsize=8, color="#2E7D32")
    for s, b in zip(steps_, bwt_vals):
        ax2.annotate(f"{b:+.3f}", xy=(s, b), xytext=(5, -12), textcoords="offset points",
                     fontsize=8, color="#8E24AA")

    kill_msg = " — TRIGGERED" if kill_state.get("triggered", False) else ""
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Prec@K (%)")
    ax.set_title(f"Forgetting monitor (N={probe_history[0]['num_clips']}, "
                 f"patience={thresholds['forgetting_patience']}, "
                 f"threshold={thresholds['forgetting_threshold_pct']:.1f} pp{kill_msg})")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper left", bbox_to_anchor=(1.15, 1.0),
              fontsize=7, framealpha=0.95, title="legend", title_fontsize=8)
    ax.grid(True, alpha=0.3)
    for ext in (".png", ".pdf"):
        plt.savefig(output_dir / f"m09_forgetting{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()


# ── Plot 3: m09_val_loss.png ────────────────────────────────────────

def plot_val_loss(output_dir: Path, probe_history: list):
    """3-panel stacked: JEPA total, masked, context — per-panel auto-scaled y."""
    if not all("val_jepa_loss" in r for r in probe_history):
        return
    steps_ = [r["global_step"] for r in probe_history]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                             gridspec_kw={"hspace": 0.15})
    panels = [
        (axes[0], "val_jepa_loss",    "JEPA total (L1)", "#C62828"),
        (axes[1], "val_masked_loss",  "masked loss",     "#1565C0"),
        (axes[2], "val_context_loss", "context loss",    "#2E7D32"),
    ]
    for ax_p, key, label, color in panels:
        vals = [r[key] for r in probe_history]
        ax_p.plot(steps_, vals, "o-", color=color, linewidth=2, markersize=6, alpha=0.95)
        seen = set()
        for r in probe_history:
            if r["stage_idx"] not in seen:
                ax_p.axvline(r["global_step"], color="gray", linestyle=":",
                             alpha=0.5, linewidth=1)
                seen.add(r["stage_idx"])
        ax_p.annotate(f"end={vals[-1]:.4f}", xy=(steps_[-1], vals[-1]),
                      xytext=(5, 0), textcoords="offset points",
                      fontsize=8, color=color, va="center")
        ax_p.set_ylabel(label, color=color, fontsize=10, fontweight="bold")
        ax_p.tick_params(axis="y", labelsize=9, colors=color)
        ax_p.grid(True, alpha=0.3)
        ax_p.margins(y=0.05)
    axes[-1].set_xlabel("Optimizer step", fontsize=10)
    fig.suptitle(f"JEPA val-loss (N={probe_history[0]['num_clips']} val-split, "
                 f"{len(probe_history)} probes) — per-panel y-axis auto-scaled",
                 fontsize=11, fontweight="bold", y=0.995)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(output_dir / f"m09_val_loss{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=None,
                        help="m09c output dir (default: outputs/<mode>/m09c_surgery)")
    parser.add_argument("--mode", default="full", choices=["sanity", "poc", "full"],
                        help="only used when --dir is not given")
    parser.add_argument("--train-config", type=Path,
                        default=Path("configs/train/ch11_surgery.yaml"),
                        help="ch11_surgery.yaml for threshold lookup")
    args = parser.parse_args()

    output_dir = args.dir or Path(f"outputs/{args.mode}/m09c_surgery")
    probe_file = output_dir / "probe_history.jsonl"
    summary_file = output_dir / "training_summary.json"

    if not probe_file.exists():
        print(f"FATAL: {probe_file} not found", file=sys.stderr)
        sys.exit(1)

    probe_history = [json.loads(line) for line in open(probe_file) if line.strip()]
    if len(probe_history) < 2:
        print(f"FATAL: need ≥2 probes, got {len(probe_history)}", file=sys.stderr)
        sys.exit(1)

    # best_state + kill_state: prefer training_summary.json (canonical);
    # fall back to max() over probe_history when training is still running.
    if summary_file.exists():
        summary = json.load(open(summary_file))
        best_state = {
            "global_step": summary["best_ckpt"]["global_step"],
            "prec_at_k":   summary["best_ckpt"]["prec_at_k"],
        }
        kill_state = {"triggered": summary["early_stop"]["triggered"]}
    else:
        best_idx = max(range(len(probe_history)),
                       key=lambda i: probe_history[i]["prec_at_k"]["mean"])
        best_state = {
            "global_step": probe_history[best_idx]["global_step"],
            "prec_at_k":   probe_history[best_idx]["prec_at_k"]["mean"],
        }
        kill_state = {"triggered": False}

    thresholds = load_thresholds(args.train_config, args.mode)

    plot_probe_trajectory(output_dir, probe_history)
    print(f"  ✅ {output_dir}/probe_trajectory.png  ({len(probe_history)} probes)")
    plot_forgetting(output_dir, probe_history, best_state, kill_state, thresholds)
    print(f"  ✅ {output_dir}/m09_forgetting.png")
    plot_val_loss(output_dir, probe_history)
    print(f"  ✅ {output_dir}/m09_val_loss.png")

    print(f"\nSource: {probe_file} · {len(probe_history)} probes · "
          f"steps {probe_history[0]['global_step']}-{probe_history[-1]['global_step']}")
    print(f"Thresholds from: {args.train_config}")
    print(f"NOTE: m09_train_loss.png is generated by utils.plots.plot_training_curves "
          f"from loss_log.csv — regenerate via a separate m09c run or direct utility call.")


if __name__ == "__main__":
    main()
