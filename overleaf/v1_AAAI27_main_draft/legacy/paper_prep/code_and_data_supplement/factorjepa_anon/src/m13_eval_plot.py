"""m13 — N-encoder visualization for the FULL probe eval suite (VISUALIZATION band, LAST).
Renamed from probe_plot.py (was m08d) — strict pipeline direction: visualization is the LAST band.

CPU-only. Pure-visualization — ALWAYS recomputes; no cache_policy.

iter16 §3.2/§3.3/§7 (2026-05-28): ALL LINE PLOTS REMOVED (§3.3a). Emits ONLY bar-with-CI panels
for ALL 14 metrics (§3.3c) + (pending) the HERO (table-with-CI + plot-with-CI). Every number
carries its BCa 95% CI. The 14 metrics (plan §7.1):
  HEAD (4): action_top1 · motion_cos · taxonomy_f1 · future_mse
  PRED (6): rollout · causal · tdist · teacher_free · maskratio · order        (m12e)
  ENC  (4): aot · tov · pace · tcc(τ)   [+ tcc_cycle appendix]                 (m12f)

Reads (auto-discovers encoders from each JSON; graceful — absent source → that metric skipped):
  <action-probe-root>/probe_paired_delta.json          {by_encoder}            → action_top1
  <motion-cos-root>/probe_motion_cos_paired.json        {by_encoder}            → motion_cos
  <future-mse-root>/probe_future_mse_per_variant.json   {by_variant}            → future_mse
  <taxonomy-root>/per_dim_acc.json                      {dims[*].by_encoder}    → taxonomy_f1
  <predictor-temporal-root>/predictor_temporal_per_variant.json {metrics}       → 6 PRED metrics
  <encoder-temporal-root>/encoder_temporal_per_variant.json     {metrics}       → 4 ENC metrics

Writes (under <output-dir>/eval/{head,predictor,encoder}/):
  one bar-with-CI panel per available metric (legacy names kept for the 3 v15a headline panels:
  probe_action_acc_compare / probe_motion_cos_compare / probe_future_mse_compare; the rest m13_*).
  (§3.3c-hero will add m13_hero_table + m13_hero_raw_values.)

USAGE:
  python -u src/m13_eval_plot.py --SANITY \\
    --action-probe-root outputs/sanity/probe_action \\
    --motion-cos-root   outputs/sanity/probe_motion_cos \\
    --future-mse-root   outputs/sanity/probe_future_mse \\
    --taxonomy-root     outputs/sanity/probe_taxonomy \\
    --predictor-temporal-root outputs/sanity/predictor_temporal \\
    --encoder-temporal-root   outputs/sanity/encoder_temporal \\
    --output-dir        outputs/sanity/probe_plot \\
    2>&1 | tee logs/m13_eval_plot_sanity.log
"""
import argparse
import json
import os
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from utils.plots import (COLORS, ENCODER_COLORS, init_style, save_fig,
                         common_exponent, exp_axis_tag, fmt_mantissa)
from utils.progress import make_pbar
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics
from utils.bootstrap import N_BOOTSTRAP
from utils.metric_names import (   # SINGLE source for the metric registry → configs/metric_names.json
    names as _load_metric_names, ordered_keys as _mn_keys,
    forest_keys as _mn_forest, direction as _mn_dir)
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)
from utils.validity import (criterion_rho, family_summary,  # iter19 §2b construct validity
                            orient_higher_better, pairwise_spearman)
from utils.config import get_pipeline_config  # iter19 §2b: plots.validity n_perm/seed (single source)
from utils.arm_registry import (arm2dir, arm2enc, display_arms, merge_arms,  # iter18: SINGLE-SOURCE arm roster (no hardcoded names)
                                merge_recipe, label_for, paper_scorecard_group)

# iter18 W7 (PLR2004): semantic named constants.
_AXIS_DECIMAL_MIN = 0.01   # tick-format band edges (display only)
_AXIS_INT_MIN = 10
_FONT_MIN_PT = 28          # hero-figure auto-shrink floor
_K_FMT_MIN = 1000          # render bootstrap count as 'N K' above this
_MIN_COMPARABLE = 2
_LABEL_ROT = 25            # x-tick names AND bar-value labels share ONE rotation (user 2026-06-15: rotate scores like the names)


# ── Display helpers (no hardcoded encoder list — derived per-call) ───

def _canon(enc: str) -> str:
    """The iter16 champion (V-JEPA 2.1 ViT-G) was evaluated under the LEGACY name vjepa_2_1_<arm>;
    rewrite it to vjepa_2_1_vitG_<arm> so §G labels it consistently with this run's vjepa_2_1_vitg /
    vjepa_2_0_vitg. Anything else (vjepa_2_1_vitg/_vitL, vjepa_2_0_*, frozen baselines) falls straight
    through unchanged — DISPLAY ONLY, never used to key into metric data."""
    if enc.startswith("vjepa_2_1_") and not enc.startswith("vjepa_2_1_vit"):
        return "vjepa_2_1_vitG_" + enc[len("vjepa_2_1_"):]
    return enc


# (backbone prefix → long tag, short tag) for the 3 trained backbones; everything else (frozen-9) falls through.
_BB_TAG = (("vjepa_2_1_vitG_", "ViT-G", "G"), ("vjepa_2_1_vitg_", "ViT-g", "g"), ("vjepa_2_0_vitg_", "ViT-g·2.0", "g0"))
# 2026-07-22: _ARM_LONG + _ARM_SHORT DELETED. They were two of FOUR competing label maps in this
# file (with _MW_TRAIN_STYLE[0] and _xb_arm_short), so one arm printed up to four different names
# across figures pages apart, and each map silently fell through for 9-17 of the 25 arms. Every
# label now comes from utils.arm_registry.label_for() → configs/arm_registry.yaml (full arm name,
# '_encoder' stripped). Derived, so it cannot drift and a new arm needs no edit here.

# Per-backbone header label (name + model size) for the stacked §G overview. Architecture facts are
# pinned in configs/model/*.yaml (embed_dim/depth VERIFIED there); mirrored here as the plot caption.
_BB_LABEL = {
    "vjepa_2_1_vitG": "V-JEPA 2.1   ·   ViT-G   ·   ~2B params   ·   1664-dim, 48 blocks   (CHAMPION)",
    "vjepa_2_1_vitg": "V-JEPA 2.1   ·   ViT-g   ·   ~1B params   ·   1408-dim, 40 blocks",
    "vjepa_2_0_vitg": "V-JEPA 2.0   ·   ViT-g   ·   ~1B params   ·   1408-dim, 40 blocks",
}


def _display_label(enc: str) -> str:
    """Human-readable encoder name for plot legends. All 3 trained backbones → '<ViT-tag> <arm>';
    frozen-9 baselines fall through verbatim. Arm name via the SINGLE source (arm_registry)."""
    enc = _canon(enc)
    for pre, long_tag, _ in _BB_TAG:
        if enc.startswith(pre):
            return f"{long_tag} {label_for(enc[len(pre):])}"
    return label_for(enc)


_FALLBACK_COLOR_CYCLE = ("blue", "green", "orange", "purple", "red", "cyan", "gold")


# iter18 2026-06-08: the 4 OURS (surgery novelty) share GREEN so they read as ONE group in every
# bar — but stay 4 separately-labelled bars (NOT merged) so you can see WHICH ours wins by height.
_OURS_GREEN = {"surgical_3stage_DI_encoder", "surgical_noDI_encoder",
               "surgical_3stage_DI_head", "surgical_noDI_head"}
# user goal (2026-06-14): EVERY variant of OUR SURGERY NOVELTY renders GREEN in the bar panels (F3 EVAL
# scorecard + the eval/ hero bars) — never a non-green hue. "Our surgery" = registry kind ∈ {surgery,
# surgery_head, merge}: this INCLUDES surgery_raw (kind=surgery) + the 4 improvement arms + the wiseft
# merges, but EXCLUDES surgical_autorgn (kind=baseline — a prior FT technique, NOT our novelty → keeps its
# own orange). Kept SEPARATE from _OURS_GREEN above on purpose: _OURS_GREEN is the WIN-CREDIT set used by the
# hero win-attribution (where raw/autorgn must NOT be credited as ours — see _is_ours below); this set is
# COLOUR-only. Single-source via registry kinds → a new surgery/improvement arm auto-greens with no edit.
_SURGERY_GREEN_ENCODERS = {e for _a, e, _g, k in display_arms(include_merge=True)
                           if k in ("surgery", "surgery_head", "merge")}
# every OTHER arm gets its OWN distinct colour, keyed by NAME (stable across all metric panels —
# the old code keyed off enc.startswith("vjepa") so ALL 14 encoders rendered in one identical blue).
# (surgery_raw is intentionally absent here — _color_for gives it its OWN light-green #81C784: surgery-family but the ablation/control, NOT the novelty green.)
_ITER18_ENC_COLOR = {
    "frozen":                   "#616161",  # gray   — frozen baseline
    "pretrain_encoder":         "#1565C0",  # blue   — vanilla cont-SSL anchor
    "surgical_autorgn_encoder": "#E65100",  # orange — Auto-RGN (Surgical-FT baseline, NOT our novelty)
    "full_ft_encoder":          "#C62828",  # red    — Full-FT
    "lpft_encoder":             "#D81B60",  # magenta— LP-FT
    "peft_lora_encoder":        "#6A1B9A",  # purple — LoRA
    "peft_dora_encoder":        "#00838F",  # cyan   — DoRA
    "cassle_encoder":           "#F9A825",  # gold   — CaSSLe
    "ewc_encoder":              "#827717",  # olive  — EWC
}
_SURG_GREEN = "#2E7D32"   # the OURS-group green, reused to flag surgery in the WINNER row
# iter18 2026-06-08: WINNER-row render style for the hero heatmap. At POC the per-metric 95% CIs are
# wide enough that a surgery arm is a CO-LEADER (CI overlaps the point-best) on EVERY metric, so a
# single point-best name under-sells it. Three honest styles, selected by env so all three can be
# rendered side-by-side for the user to pick (none invents significance — every style keeps the "~"/
# tie marker and boxes ALL CI-tied co-leaders):
#   coleader_set     — list ALL arms tied for #1 (surgery bold-green), competitors shown too
#   surgery_coleader — name the best SURGERY arm (green); "~" prefix when it's a CI-tie
#   tie_badge        — "~ tie (n)" / "<arm> WIN (sole)"; names nobody, boxes carry the co-leaders
_WINNER_MODE = os.environ.get("HERO_WINNER_MODE", "coleader_set")


def _color_for(enc: str, idx: int) -> str:
    """Per-encoder colour: EVERY surgery-novelty arm → one GREEN; every other arm → its own distinct colour
    keyed by name (stable across panels). Legacy/cross-arch encoders fall back to the canonical map."""
    # Normalize to the arm-encoder token for ANY trained backbone, so the 1B (vjepa_2_1_vitg_<arm>) and the
    # 2B champion (legacy vjepa_2_1_<arm> → vjepa_2_1_vitG_<arm> via _canon) key into the SAME colour maps.
    # 2026-06-24: before this, the 1B's surviving "vitg_" prefix missed every map → index fallback (all bars
    # got arbitrary colours), while the 2B matched. Now both reduce identically → 2B output is unchanged.
    short = _canon(enc)
    for _pre, *_rest in _BB_TAG:
        if short.startswith(_pre):
            short = short[len(_pre):]
            break
    else:
        short = enc.replace("vjepa_2_1_", "")            # frozen-9 / dinov2 / non-backbone names: legacy strip
    # surgery_raw = the surgery TECHNIQUE on RAW clips (no factorization) — OUR ablation/control, a
    # COMPETITOR (group=surgery_ablation → already OTHER in the verdict _MW_FAM_OURS). LIGHT green: reads
    # as "surgery family" yet visibly distinct from the novelty dark-green → "a surgery variant, but the
    # ablation" (user 2026-06-23). MUST precede the green check below (which would otherwise dark-green it).
    if short == "surgery_raw_encoder":
        return "#81C784"                                 # LIGHT green — surgery-on-raw ablation/control
    if short in _OURS_GREEN or short in _SURGERY_GREEN_ENCODERS:
        return "#2E7D32"                                  # GREEN — our surgery novelty (factor surgery + improvement + wiseft)
    if short in _ITER18_ENC_COLOR:
        return _ITER18_ENC_COLOR[short]
    if enc in ENCODER_COLORS:
        return ENCODER_COLORS[enc]
    return COLORS.get(_FALLBACK_COLOR_CYCLE[idx % len(_FALLBACK_COLOR_CYCLE)], COLORS["gray"])


# ── Loaders ──────────────────────────────────────────────────────────

def _load_json(path: Path, stage_hint: str) -> dict:
    """REQUIRED JSON — FAIL LOUD if missing (CLAUDE.md: no silent defaults)."""
    if not path.exists():
        sys.exit(f"FATAL: {path} not found — run {stage_hint} first")
    return json.loads(path.read_text())


def _opt_json(path: Path):
    """OPTIONAL JSON — returns None (graceful) if absent. For the newer temporal/taxonomy
    sources that a SKIP_STAGES run may not have produced; m13 degrades to fewer metrics."""
    return json.loads(path.read_text()) if path.exists() else None


# ── Bar-with-CI primitive (N-bar generic) ───────────────────────────

def _axis_labels(encoders: list) -> list:
    """x-tick names for a bar panel, with the backbone tag dropped when EVERY bar shares it.

    The tag already appears in the figure's own banner (_BB_LABEL: "V-JEPA 2.1 · ViT-g · ~1B …"),
    so repeating "ViT-g " on all 24 ticks is pure redundancy: it steals label width and forces the
    45°-rotated names to overlap into an unreadable smear (2026-07-22). A panel mixing backbones —
    or the frozen-9 chart, where names like "DINOv2 frozen" share no tag — keeps the full name, so
    the disambiguation is never lost where it actually matters."""
    names = [_display_label(e) for e in encoders]
    if not names:
        return names
    known = {t for _pre, t, _short in _BB_TAG}
    heads = {n.split(" ", 1)[0] for n in names}
    if len(heads) == 1 and heads <= known and all(" " in n for n in names):
        cut = len(next(iter(heads))) + 1
        return [n[cut:] for n in names]
    return names


def _bar_with_ci(ax, encoders: list, vals: list, errs: list,
                 ylabel: str, title: str, na_set: set = None, direction: str = ""):
    """Render N bars with 95% CI error caps + value labels. `direction`: higher/lower → badge;
    "" → none. `na_set` = encoders with no measurement → hatched 'N/A'."""
    na_set = na_set or set()
    x = np.arange(len(encoders))
    colors = [_color_for(e, i) for i, e in enumerate(encoders)]
    plot_vals = [0.0 if e in na_set else v for e, v in zip(encoders, vals)]
    plot_errs = [0.0 if e in na_set else er for e, er in zip(encoders, errs)]
    # iter18 (2026-06-13, user order): auto common-exponent so clustered small decimals
    # (e.g. teacher-free raw/lora/ours all "0.045") separate — rescale bars + value labels
    # and carry a "(×10⁻ⁿ)" tag on the y-axis label. Real (non-N/A) values set the exponent.
    scale, exp = common_exponent(
        [v for e, v in zip(encoders, plot_vals) if e not in na_set],
        [er for e, er in zip(encoders, plot_errs) if e not in na_set])
    plot_vals = [v * scale for v in plot_vals]
    plot_errs = [(er if (isinstance(er, float) and np.isnan(er)) else er * scale)
                 for er in plot_errs]
    bars = ax.bar(x, plot_vals, 0.6, color=colors, alpha=0.85,
                  yerr=plot_errs, capsize=4, error_kw={"lw": 1.2, "ecolor": "#222"})
    for i, e in enumerate(encoders):
        if e in na_set:
            bars[i].set_hatch("//")
            bars[i].set_alpha(0.25)
    real_v = np.array([v for e, v in zip(encoders, plot_vals) if e not in na_set])
    real_e = np.array([er for e, er in zip(encoders, plot_errs) if e not in na_set])
    if real_v.size:
        real_e_safe = np.nan_to_num(real_e, nan=0.0)
        lo = float((real_v - real_e_safe).min())
        hi = float((real_v + real_e_safe).max())
        span = hi - lo
        pad = max(0.15 * span, 0.02 * abs(hi)) if span > 0 else (abs(hi) * 0.1 or 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        pad = 0.05
    y_lo, y_hi = ax.get_ylim()
    # Lean value labels toward the SHORT-bar side so they fall into empty space, not onto the next bar:
    # higher=better sorts tall→short L→R → lean +rot (toward the short RIGHT); lower=better sorts short→tall
    # → lean −rot (toward the short LEFT). The x-tick names below the axis keep +rot (user 2026-06-15).
    _val_rot = -_LABEL_ROT if direction == "lower" else _LABEL_ROT
    for xi, e, v, er in zip(x, encoders, plot_vals, plot_errs):
        if e in na_set:
            ax.text(xi, y_lo + (y_hi - y_lo) * 0.5, "N/A", ha="center", va="center",
                    fontsize=12, color="#555", fontweight="bold")
        else:
            er_safe = 0.0 if (isinstance(er, float) and np.isnan(er)) else er
            ax.text(xi, v + er_safe + (y_hi - y_lo) * 0.01, fmt_mantissa(v),
                    ha="center", va="bottom", fontsize=9, color="#222", rotation=_val_rot)
    ax.set_xticks(x)
    ax.set_xticklabels(_axis_labels(encoders), fontsize=9, rotation=_LABEL_ROT, ha="right")
    # analysis-phase aid (user 2026-06-15): tint each x-label with ITS bar's colour so a name maps to its bar at
    # a glance (NOT paper-acceptable — paper keeps black; fine for these analysis-phase scorecards).
    for _tick, _c in zip(ax.get_xticklabels(), colors):
        _tick.set_color(_c)
    ax.set_ylabel(ylabel + exp_axis_tag(exp), fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if direction == "higher":
        # higher=better sorts tallest-bar-first (LEFT) → a top-LEFT badge overlaps the bars; put it top-RIGHT
        # (the short-bar side). lower=better is the mirror image, so it STAYS top-left (user 2026-06-15).
        ax.text(0.98, 0.97, "↑ higher = better", transform=ax.transAxes, fontsize=10,
                fontweight="bold", color="#2E7D32", va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                          edgecolor="#2E7D32", linewidth=1.0, alpha=0.85))
    elif direction == "lower":
        ax.text(0.02, 0.97, "↓ lower = better", transform=ax.transAxes, fontsize=10,
                fontweight="bold", color="#E65100", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0",
                          edgecolor="#E65100", linewidth=1.0, alpha=0.85))
    elif direction == "signed":
        # signed = a DIAGNOSTIC with no "better" direction (e.g. frame-order sensitivity) — neutral badge
        ax.text(0.98, 0.97, "± signed (diagnostic)", transform=ax.transAxes, fontsize=10,
                fontweight="bold", color="#6A1B9A", va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5",
                          edgecolor="#6A1B9A", linewidth=1.0, alpha=0.85))


def _sort_by_metric(encoders: list, vals: list, errs: list, na_set: set, direction: str):
    """Sort (encoder, val, err) by val; N/A always tail. direction 'desc' (higher=better) /
    'asc' (lower=better). NaN treated as N/A."""
    triples = list(zip(encoders, vals, errs))
    real = [(e, v, er) for e, v, er in triples
            if e not in na_set and not (isinstance(v, float) and np.isnan(v))]
    na = [(e, v, er) for e, v, er in triples
          if e in na_set or (isinstance(v, float) and np.isnan(v))]
    real.sort(key=lambda t: t[1], reverse=(direction == "desc"))
    out = real + na
    return [t[0] for t in out], [t[1] for t in out], [t[2] for t in out]


def _emit_bar(out_path: Path, sorted_enc, sorted_vals, sorted_errs, na_set,
              ylabel, title, badge, caption, layman, n_enc, boot_str):
    """One bar-with-CI figure (png+pdf). Module-level (was a closure) so the 14-metric loop
    and the legacy 3 headline panels share one code path.

    iter16 fix: the caption lives in FIGURE coords at the bottom (not ax.text at a large
    negative axes-fraction, which fed back into tight_layout and collapsed the axes into a
    thin strip). Fixed margins via subplots_adjust give a TALL plot area (~1.5:1), matching
    the iter15 reference. Caption now carries a plain-language `Layman:` example."""
    fig, ax = plt.subplots(figsize=(max(8.5, 0.95 * n_enc + 3.0), 9.5))
    _bar_with_ci(ax, sorted_enc, sorted_vals, sorted_errs,
                 ylabel=ylabel, title=title, na_set=na_set, direction=badge)
    cap = (f"{caption}\nLayman: e.g. {layman}\n"
           f"TEST split · whiskers = BCa 95 % CI from {boot_str}.")
    fig.text(0.5, 0.03, cap, ha="center", va="bottom", fontsize=10, color="#000",
             linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA", edgecolor="#888", linewidth=0.8))
    fig.suptitle(f"{title} · {n_enc} encoders · 95 % BCa CI", fontsize=14, fontweight="bold", y=0.985)
    # Fixed margins: TALL axes (top-bottom = 0.59 of a 9.5in fig ≈ 5.6in) + room for rotated
    # xticklabels (bottom 0.33) and the caption box (figure y≈0.03).
    fig.subplots_adjust(left=0.11, right=0.96, top=0.92, bottom=0.33)
    save_fig(fig, str(out_path))


# ── §3.3c: 14-metric catalog + unified loader + per-metric bars (plan §7) ──
# (key, family, out_name, direction, ylabel, caption, layman). direction: higher|lower|signed.
# `layman` = one-line plain-language example shown in every panel's caption box (iter16 fix).
# Legacy names kept for the 3 v15a headline panels; the rest take m13_ names. Mirrors §7.1.
_CATALOG = [
    ("action_top1",  "HEAD", "probe_action_acc_compare", "higher", "Top-1 accuracy (%)",
     "AttentiveClassifier → K-class motion-flow accuracy (magnitude × direction).",
     "from the clip's features alone, can a simple classifier tell 'fast leftward motion' from 'slow upward motion'?"),
    ("motion_cos",   "HEAD", "probe_motion_cos_compare", "higher", "Intra − Inter cosine",
     "Same-class minus different-class cosine separation (>0 ⇒ motion-semantic clustering).",
     "do two clips of the SAME motion sit closer in feature space than two clips of DIFFERENT motions?"),
    ("taxonomy_f1",  "HEAD", "m13_taxonomy_f1_compare",  "higher", "Taxonomy mean (top1+F1)",
     "Mean over 15 VLM-tag dims of per-dim top-1 (single) / sample-F1 (multi).",
     "can a linear read-off name scene attributes — crowd density, lighting, camera motion — from the features?"),
    ("future_mse",   "HEAD", "probe_future_mse_compare", "lower",  "Future L1",
     "V-JEPA predictor L1 on masked next-frame tokens (the JEPA objective; V-JEPA-only).",
     "given the first frames, how well does the model's predictor guess the next frame's content?"),
    ("rollout",      "PRED", "m13_rollout_compare",      "lower",  "Free-run drift slope",
     "Iterated multi-step rollout L1 slope — lower = stabler free-running dynamics.",
     "if the model predicts step-by-step into the future, how fast does its error snowball?"),
    ("causal",       "PRED", "m13_causal_compare",       "lower",  "Causal future-block L1",
     "Predict future temporal half from past half — lower = better causal prediction.",
     "shown only the first half of a clip, how well does it predict the second half?"),
    ("tdist",        "PRED", "m13_tdist_compare",        "lower",  "L1-vs-Δt slope",
     "Single-shot L1 vs target offset Δt — lower = slower decay with temporal distance.",
     "does prediction stay accurate as you ask it to look further ahead in time?"),
    ("teacher_free", "PRED", "m13_teacher_free_compare", "lower",  "Free − teacher gap",
     "Free-running minus teacher-forced L1 — lower = less exposure bias.",
     "how much worse is the model when it must rely on its OWN predictions vs being fed the truth each step?"),
    ("maskratio",    "PRED", "m13_maskratio_compare",    "lower",  "L1-vs-maskratio slope",
     "L1 vs mask-ratio sweep — lower = graceful degradation under heavier masking.",
     "how gracefully does prediction degrade as more of the video is hidden?"),
    ("order",        "PRED", "m13_order_compare",        "signed", "Shuffled − ordered ΔL1",
     "Shuffled-context minus ordered-context L1 — sign = predictor's reliance on frame order.",
     "does scrambling the frame order hurt the model — i.e. does it actually use time?"),
    ("aot",          "ENC",  "m13_aot_compare",          "higher", "Arrow-of-Time acc",
     "Forward-vs-reversed binary head accuracy — higher = encoder preserves time's arrow.",
     "can the model tell a video played forwards from the same video played backwards?"),
    ("tov",          "ENC",  "m13_tov_compare",          "higher", "TOV / VCOP top-1",
     "N-way frame-permutation classification top-1 — higher = retains temporal order.",
     "can it put a handful of shuffled frames back into the right order?"),
    ("pace",         "ENC",  "m13_pace_compare",         "higher", "Pace top-1",
     "Playback-rate (1×/2×/4×) classification top-1 — higher = temporal-scale sensitive.",
     "can it tell normal-speed video from 2× / 4× fast-forwarded video?"),
    ("tcc_tau",      "ENC",  "m13_tcc_tau_compare",      "higher", "TCC Kendall's τ",
     "Per-frame soft-NN alignment Kendall's τ across same-action pairs (training-free).",
     "do matching moments in two clips of the same action line up in feature space?"),
    ("tcc_cycle",    "ENC",  "m13_tcc_cycle_compare",    "lower",  "TCC cycle-back error",
     "TCC A→B→A cycle-back frame-index error (appendix companion to τ) — lower = better.",
     "follow a frame to its match in clip B and back — how far from where it started does it land?"),
]
_FAMILY_DIR = {"HEAD": "head", "PRED": "predictor", "ENC": "encoder"}
_DIR_TAG = {"higher": "↑ better", "lower": "↓ better", "signed": "± signed"}   # column direction badge


def _short_label(enc: str) -> str:
    """Backbone-tagged encoder tag for the hero WINNER row. '<g-tag>-<arm>' for the 3 backbones.
    Same arm name as every other figure (arm_registry) — only the backbone tag is abbreviated, so a
    cell can never disagree with the axis label beside it."""
    enc = _canon(enc)
    for pre, _, short_tag in _BB_TAG:
        if enc.startswith(pre):
            return f"{short_tag}-{label_for(enc[len(pre):])}"
    return label_for(enc)


def _fmt_compact(x: float) -> str:
    """Sign-prefixed number ROUNDED TO 2 decimals (0.01) for heatmap cells. Anything that rounds to
    zero prints '0' (so −0.0003 → 0, −0.07 stays −0.07, +5.81 stays +5.81)."""
    r = round(float(x), 2)
    if r == 0:                                # also catches -0.0 (e.g. round(-0.0003, 2))
        return "0"
    return f"{r:+.2f}"


def _fmt_fine(x: float) -> str:
    """Adaptive-precision sign-prefixed number for the PAIRED-diff heatmap, where the values are tiny
    (predictor Δs ~1e-3) — 0.01 rounding would collapse a significant Δ to a misleading '0'. Keeps 2–4
    sig decimals so a coloured 'SURGERY' cell never reads as 0."""
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    ax = abs(x)
    if ax >= 1:
        return f"{x:+.2f}"
    if ax >= _AXIS_DECIMAL_MIN:
        return f"{x:+.3f}"
    if ax > 0:
        return f"{x:+.4f}"
    return "0"


def _fmt_val(x: float) -> str:
    """Adaptive-precision RAW metric value (NO forced sign — these are absolute values, not deltas).
    Raw metrics span ~0.009 (predictor) → ~53 (action top1), so a fixed 0.01 rounding would collapse the
    tiny ones; big numbers stay terse (1 decimal), small ones get the precision they need."""
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    ax = abs(x)
    if ax >= _AXIS_INT_MIN:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.2f}"
    if ax >= _AXIS_DECIMAL_MIN:
        return f"{x:.3f}"
    return f"{x:.4f}"


def _taxonomy_f1_by_encoder(taxonomy_json: dict, encoders: list) -> dict:
    """taxonomy_f1 per encoder = mean over the 15 dims of per-dim test_mean; ci = mean of the
    per-dim ci_half (a conservative aggregate of the per-dim BCa half-widths)."""
    dims = taxonomy_json.get("dims", {})
    out = {}
    for v in encoders:
        vals, cis = [], []
        for spec in dims.values():
            be = spec.get("by_encoder", {}).get(v)
            if be is not None:
                vals.append(be["test_mean"])
                cis.append(be["test_ci"]["ci_half"])
        if vals:
            # nan-filter the per-dim CI halves (a degenerate per-dim BCa → nan; don't poison the
            # aggregate — tiny SANITY splits trip this, real POC/FULL won't).
            clean = [c for c in cis if not (isinstance(c, float) and np.isnan(c))]
            out[v] = (float(np.mean(vals)), float(np.mean(clean)) if clean else 0.0)
    return out


_DM = ("delta_mean", "delta_ci_lo", "delta_ci_hi")   # motion/future/pred/enc/tcc pairwise schema


def _norm_deltas(pairwise: dict, fields: tuple) -> dict:
    """Normalize a pairwise_deltas block → {pair_key: (delta, lo, hi)} using this schema's
    (delta, lo, hi) field names. pair_key is the JSON's '{a}_minus_{b}' (= a − b)."""
    fd, fl, fh = fields
    return {k: (v[fd], v[fl], v[fh]) for k, v in pairwise.items()}


def _agg_taxonomy_deltas(taxonomy_json: dict) -> dict:
    """taxonomy aggregate Δ per pair = mean over the 15 dims of per-dim (delta, ci_lo, ci_hi)."""
    acc = {}
    for spec in taxonomy_json.get("dims", {}).values():
        for pk, v in spec.get("pairwise_deltas", {}).items():
            acc.setdefault(pk, []).append((v["delta"], v["ci_lo"], v["ci_hi"]))
    return {pk: tuple(float(np.mean(col)) for col in zip(*rows)) for pk, rows in acc.items()}


def _load_all_metrics(srcs: dict, encoders: list) -> dict:
    """Return {metric_key: {"by_encoder": {enc: (val, ci_half)}, "na": set, "deltas": {pair_key:
    (Δ, lo, hi)}}} for every catalog metric whose source JSON is present (graceful: absent → omitted).
    `deltas` (the paired Δ across variants) feeds the hero's Δ-vs-frozen + CI-exclusion *."""
    out = {}

    def _pack(by_enc, deltas):
        return {"by_encoder": by_enc, "na": {e for e in encoders if e not in by_enc}, "deltas": deltas}

    if srcs.get("action"):
        be = srcs["action"].get("by_encoder", {})
        out["action_top1"] = _pack(
            {v: (be[v]["acc_pct"], be[v]["top1_ci"]["ci_half"] * 100.0) for v in be},
            _norm_deltas(srcs["action"].get("pairwise_deltas", {}), ("delta_pp", "ci_lo_pp", "ci_hi_pp")))
    if srcs.get("motion"):
        be = srcs["motion"].get("by_encoder", {})
        out["motion_cos"] = _pack(
            {v: (be[v]["score_mean"], be[v]["score_ci"]["ci_half"]) for v in be},
            _norm_deltas(srcs["motion"].get("pairwise_deltas", {}), _DM))
    if srcs.get("future"):
        bv = srcs["future"].get("by_variant", {})
        real = {v: e for v, e in bv.items() if isinstance(e, dict)}
        out["future_mse"] = _pack(
            {v: (real[v]["mse_mean"], real[v]["mse_ci"]["ci_half"]) for v in real},
            _norm_deltas(srcs["future"].get("pairwise_deltas", {}), _DM))
    if srcs.get("taxonomy"):
        out["taxonomy_f1"] = _pack(_taxonomy_f1_by_encoder(srcs["taxonomy"], encoders),
                                   _agg_taxonomy_deltas(srcs["taxonomy"]))
    if srcs.get("pred"):
        m = srcs["pred"].get("metrics", {})
        for key in ("rollout", "causal", "tdist", "teacher_free", "maskratio", "order"):
            blk = m.get(key, {})
            bv = blk.get("by_variant", {})
            if bv:
                out[key] = _pack({v: (bv[v]["mean"], bv[v]["ci"]["ci_half"]) for v in bv},
                                 _norm_deltas(blk.get("pairwise_deltas", {}), _DM))
    if srcs.get("enc"):
        m = srcs["enc"].get("metrics", {})
        for key in ("aot", "tov", "pace"):
            blk = m.get(key, {})
            bv = blk.get("by_variant", {})
            if bv:
                out[key] = _pack({v: (bv[v]["mean"], bv[v]["ci"]["ci_half"]) for v in bv},
                                 _norm_deltas(blk.get("pairwise_deltas", {}), _DM))
        tcc = m.get("tcc", {})
        tcc_bv = tcc.get("by_variant", {})
        if tcc_bv:
            tpd = tcc.get("pairwise_deltas", {})
            out["tcc_tau"] = _pack(
                {v: (tcc_bv[v]["kendalls_tau"]["mean"], tcc_bv[v]["kendalls_tau"]["ci"]["ci_half"]) for v in tcc_bv},
                _norm_deltas(tpd.get("kendalls_tau", {}), _DM))
            out["tcc_cycle"] = _pack(
                {v: (tcc_bv[v]["cycle_back"]["mean"], tcc_bv[v]["cycle_back"]["ci"]["ci_half"]) for v in tcc_bv},
                _norm_deltas(tpd.get("cycle_back", {}), _DM))
    return out


# ── §3.3c HERO (B1 table-with-CI + B2 Δ-vs-frozen heatmap) — plan §7.3 ──

def _delta_v_vs_frozen(deltas: dict, v: str, frozen: str):
    """Δ(v − frozen), (lo, hi) from a normalized deltas dict keyed '{a}_minus_{b}' (=a−b).
    Orients + sign-flips (and swaps CI bounds) when stored as frozen−v. None if absent."""
    kvf, kfv = f"{v}_minus_{frozen}", f"{frozen}_minus_{v}"
    if kvf in deltas:
        d, lo, hi = deltas[kvf]
        return d, lo, hi
    if kfv in deltas:
        d, lo, hi = deltas[kfv]
        return -d, -hi, -lo          # v−f = −(f−v); CI [lo,hi] → [−hi,−lo]
    return None


def _ci_excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def _is_good_win(delta: float, lo: float, hi: float, direction: str) -> bool:
    """Significant win vs frozen in the GOOD direction. higher→lo>0 ; lower→hi<0 ; signed→never."""
    if direction == "higher":
        return lo > 0
    if direction == "lower":
        return hi < 0
    return False


def _hero_catalog(metrics: dict):
    """The 14 hero metrics present (tcc_cycle is the appendix companion — excluded from the grid)."""
    return [c for c in _CATALOG if c[0] in metrics and c[0] != "tcc_cycle"]


def plot_hero_table(metrics: dict, encoders: list, frozen: str, output_dir: Path, boot_str: str):
    """B1: value±CI scorecard PNG + CSV, rendered as a COLORED heatmap-table.
    rows = encoders (frozen pinned top) + a WINNER row ; cols = hero metrics + WINS.
      • cell colour  = per-column min-max normalized, direction-aware (green=best in that metric);
                       'order' (signed) + N/A → neutral.
      • '*'          = Δ-vs-frozen 95% BCa CI excludes 0 (significance vs baseline).
      • winner       = single best encoder per scorable metric (argmax/argmin); its cell is BOLD,
                       named in the WINNER row.
      • WINS         = #metrics this encoder is THE winner, of n_scorable → the column PARTITIONS
                       the metrics (Σ over all rows = n_scorable; iter16 fix — was an overlapping
                       'beats-frozen' count that summed to >1)."""
    cat = _hero_catalog(metrics)
    if not cat:
        print("  [hero-table] no metrics present — skip")
        return
    scorable = [c for c in cat if c[3] in ("higher", "lower")]
    ordered = [frozen] + [e for e in encoders if e != frozen]
    metric_keys = [c[0] for c in cat]
    cmap = plt.get_cmap("RdYlGn")

    # raw point values for colouring + winner determination
    valmat = {(e, k): (metrics[k]["by_encoder"].get(e, (None, None))[0]) for e in ordered for k in metric_keys}
    colstat = {}                                   # key -> (vmin, vmax) over present encoders
    for k in metric_keys:
        present = [valmat[(e, k)] for e in ordered if valmat[(e, k)] is not None]
        colstat[k] = (min(present), max(present)) if present else (0.0, 0.0)
    # WINNER = canonical surgery-vs-pretrain CHAMPION DUEL (same _family_verdict the heatmap WINNER
    # row / scoreboard / grouped use) — NOT a raw best-encoder argmax (which never ties and could
    # even crown the frozen baseline). Decisive metric → the winning arm; paired CI overlaps 0 → tie
    # (no winner, no blue box). Keeps all four §G views telling ONE story.
    _surg = [e for e in ordered if _arm_family(e) == "surgery"]
    _pre = [e for e in ordered if _arm_family(e) == "pretrain"]
    _per = _family_verdict(metrics, encoders, frozen)[3]
    winner = {}
    for key, _f, _o, direction, _y, _c, _l in scorable:
        v = _per.get(key)
        if v == "surgery":
            winner[key] = _family_champion(metrics, key, frozen, _surg)
        elif v == "pretrain":
            winner[key] = _family_champion(metrics, key, frozen, _pre)
        # v == "tie" (or absent) → no winner entry → WINNER col shows "tie", no blue box

    def _cell_colour(val, key, direction):
        if val is None:
            return (0.90, 0.90, 0.90, 1.0)                       # N/A grey
        if direction not in ("higher", "lower"):
            return (1.0, 1.0, 1.0, 1.0)                          # signed → neutral
        vmin, vmax = colstat[key]
        t = 0.5 if vmax == vmin else (val - vmin) / (vmax - vmin)
        if direction == "lower":
            t = 1.0 - t                                          # low value = good = green
        r, g, b, _ = cmap(t)
        return (r, g, b, 0.55)                                   # alpha keeps black text legible

    # TRANSPOSED (vertical, single-column-paper fit): rows = metrics (+ WINS row); cols = encoders
    # (short labels) + a WINNER col. Taller-than-wide → fits one paper column far better than the
    # old 11-wide layout.
    text_rows, colour_rows, winner_coords = [], [], []
    csv_rows = [["metric"] + [_short_label(e) for e in ordered] + ["WINNER"]]
    for (key, _f, _o, direction, _y, _c, _l) in cat:
        cells, ccols, wins_ci = [], [], []
        for ci, e in enumerate(ordered):
            be = metrics[key]["by_encoder"].get(e)
            if be is None:
                cells.append("N/A"); ccols.append(_cell_colour(None, key, direction)); continue
            val, ciw = be
            star = ""
            if e != frozen:
                dvf = _delta_v_vs_frozen(metrics[key]["deltas"], e, frozen)
                if dvf and _ci_excludes_zero(dvf[1], dvf[2]):
                    star = "*"
            cells.append(f"{val:.3f}\n±{ciw:.3f}{star}")            # value + CI wrapped to 2 lines
            ccols.append(_cell_colour(val, key, direction))
            if winner.get(key) == e:
                wins_ci.append(ci)
        win_cell = (_wrap_name(winner[key]) if winner.get(key)
                    else ("tie" if direction in ("higher", "lower") else "—"))  # dead-heat duel → tie
        csv_rows.append([key] + cells + [win_cell])                # CSV keeps ALL metrics incl signed 'order'
        if direction not in ("higher", "lower"):
            continue                                               # signed (order): diagnostic-only → CSV, not drawn
        ri = len(text_rows)                                        # figure row index (scorable rows only)
        winner_coords.extend((ri, ci) for ci in wins_ci)
        cells.append(win_cell); ccols.append((1.0, 0.96, 0.78, 1.0))
        text_rows.append(cells); colour_rows.append(ccols)

    row_labels = [f"{c[0]} {_DIR_TAG[c[3]]}" for c in scorable]
    col_labels = [_short_label(e) for e in ordered] + ["WINNER"]
    fig, ax = plt.subplots(figsize=(2.0 + 1.05 * len(col_labels), 1.8 + 0.62 * len(row_labels)))
    ax.axis("off")
    tbl = ax.table(cellText=text_rows, cellColours=colour_rows,
                   rowLabels=row_labels, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 2.0)
    for (ri, ci) in winner_coords:                              # spotlight winning cells (+1 for header row):
        cell = tbl[(ri + 1, ci)]                                # bold + thick blue outline (font unchanged → CI stays readable)
        cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#1a3cff")
        cell.set_linewidth(3.0)
    ax.set_title("HERO scorecard (vertical) — value ± BCa 95% CI · colour = per-metric min-max "
                 "(green = best) · * = Δ-vs-frozen CI excludes 0\n"
                 f"{len(scorable)} metrics × {len(ordered)} encoders · WINNER col + BLUE BOX = "
                 f"surgery-vs-vanilla-cont-SSL champion duel (tie = paired 95% CI overlaps 0) · {boot_str}",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_hero_table"))
    import csv
    with open(str(output_dir / artifact("m13_hero_table")), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"  [hero-table] m13_hero_table.{{png,pdf,csv}} — {len(scorable)} metric rows drawn "
          f"(+signed 'order' in CSV only) × {len(col_labels)} cols (transposed) · WINNER col = champion duel (ties shown)")


# ── iter18 WiSE-FT v2 sweep readout (intervene × frozen predictor-merge · frozen-fraction sweep) ──
# The eval_metrics.json metric keys the prediction↔coherence trade-off is told through (same key space as
# the metrics_watch dumps; directions mirror _CATALOG). Roster is DERIVED from the registry, not listed here.
_WISEFT_SWEEP_COLS = [
    ("fut",     "Future-MSE\n↓ predict",   "lo"),
    ("causal",  "Causal-L1\n↓ predict",    "lo"),
    ("tcc_tau", "TCC τ\n↑ cohere",          "hi"),
    ("aot",     "Arrow-of-Time\n↑ cohere",  "hi"),
]


def _wiseft_sweep_rows(present: list) -> list:
    """(eval_encoder_name, display_label) for the predictor-merged WiSE-FT sweep + its base + frozen — all
    DERIVED from configs/arm_registry.yaml (merge arms with merge_predictor=true, ordered by frozen-fraction
    = 1−merge_alpha) and matched to the encoders actually present in eval_metrics.json. NO hardcoded names."""
    t2e = {a: e for a, e, _g, _k in display_arms(include_merge=True)}    # train-name → encoder token

    def _find(token):                                   # eval name (<backbone-prefix>_<token>) present in the json
        hits = [e for e in present if e == token or e.endswith("_" + token)]
        return hits[0] if hits else None

    sweep, base_arm = [], None
    for a in sorted(merge_arms()):
        rec = merge_recipe(a)
        if rec["predictor"]:                            # the predictor-merged sweep (intervene × frozen, f30/f50/f70)
            sweep.append((t2e[a], rec["alpha"]))
            base_arm = rec["base"]
    sweep.sort(key=lambda t: -t[1])                     # alpha desc → frozen-fraction asc
    out = []
    fr = _find("frozen")
    if fr:
        out.append((fr, "Frozen  (coherence reference)"))
    if base_arm:
        b = _find(t2e[base_arm])
        if b:
            out.append((b, f"{_display_label(b)}  (base · no merge)"))
    for tok, alpha in sweep:
        e = _find(tok)
        if e:
            out.append((e, f"WiSE-FT  ·  frozen {1.0 - alpha:.2f}"))
    return out


def plot_wiseft_sweep_table(metrics_json: Path, out_dir: Path, stem: str = "wiseft_sweep_table"):
    """iter18 WiSE-FT v2 sweep readout — a matplotlib value±CI TABLE (NOT ASCII). Rows (registry-derived) =
    frozen ref + the base intervene arm + the 3 frozen-fraction merges; cols = the PREDICTION pair
    (future-MSE, causal-L1) and the COHERENCE pair (TCC τ, AoT) the sweep trades between. Cell colour =
    per-column min-max, direction-aware (green = best). Any sweep arm still evaling → 'pending' cell, so this
    is safe to regenerate repeatedly as the run fills the aggregated eval_metrics.json in."""
    if not metrics_json.exists():
        print(f"  [wiseft-sweep] {metrics_json} absent — skip"); return
    data = {d["encoder"]: d for d in json.loads(metrics_json.read_text())}
    present = _wiseft_sweep_rows(list(data))
    if len(present) < 2:
        print(f"  [wiseft-sweep] <2 sweep arms present in {metrics_json.name} — skip"); return
    cmap = plt.get_cmap("RdYlGn")
    colstat = {}
    for key, _h, _d in _WISEFT_SWEEP_COLS:
        vals = [data[e][key]["mean"] for e, _ in present
                if data[e].get(key) and data[e][key]["mean"] is not None]
        colstat[key] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    text_rows, colour_rows = [], []
    for enc, _disp in present:
        cells, ccols = [], []
        for key, _h, direction in _WISEFT_SWEEP_COLS:
            cell = data[enc].get(key)
            mean = cell["mean"] if cell else None
            if mean is None:
                cells.append("—\n(pending)"); ccols.append((0.92, 0.92, 0.92, 1.0)); continue
            ci = cell["ci_half"] or 0.0
            cells.append(f"{mean:.4f}\n±{ci:.4f}")
            vmin, vmax = colstat[key]
            t = 0.5 if vmax == vmin else (mean - vmin) / (vmax - vmin)
            if direction == "lo":
                t = 1.0 - t                                      # low value = good = green
            r, g, b, _ = cmap(t)
            ccols.append((r, g, b, 0.55))
        text_rows.append(cells); colour_rows.append(ccols)
    fig, ax = plt.subplots(figsize=(3.0 + 2.1 * len(_WISEFT_SWEEP_COLS), 1.6 + 0.95 * len(present)))
    ax.axis("off")
    tbl = ax.table(cellText=text_rows, cellColours=colour_rows,
                   rowLabels=[d for _, d in present],
                   colLabels=[h for _k, h, _d in _WISEFT_SWEEP_COLS], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 2.7)
    # base arm named via the SINGLE source, so this title cannot drift from the row labels beneath it
    ax.set_title(f"WiSE-FT v2 sweep — {label_for('surgery_3stage_DI_intervene_encoder')} × frozen, "
                 "predictor merged   ·   value ± BCa 95% CI\n"
                 "prediction pair (←)  vs  coherence pair (→):  more frozen recovers coherence, gives back "
                 "prediction · colour = per-column best (green)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, str(out_dir / stem))
    print(f"  [wiseft-sweep] {stem}.{{png,pdf}} — {len(present)} sweep arms (registry-derived)")


# ═══ HONEST paper scorecard (folded from scripts/iter18_paper_scorecard.py, iter18 2026-06-14) ═══════════
# The figure that survives a hostile review: FULL baseline set (nothing hidden), the surgery FAMILY shares
# one colour so the reader SEES it cluster at the top of the predictive metrics, HERO = causal-L1 + future-MSE.
# Reads the same eval_metrics.json the watch refresh emits. Helpers/constants are _ps_*-prefixed so they don't
# collide with this module's own (or the tcc fold's) names. Figure recipe kept byte-equivalent to the original
# (direct fig.savefig png@150 + pdf, NOT save_fig — that would change dpi/facecolor).
# arm taxonomy (display short-name, group) — group drives colour/role. groups:
#   ours_flagship · surgery_ablation · ft_baseline · frozen · improvement
# 2026-07-22: the hand-written {encoder: (label, group)} dict that used to live here is GONE. It was
# the 5th of six competing label sources ("continual-SSL", "Surgery 3-stage-DI  (ours)", …) AND it
# restated a group already recorded in configs/arm_registry.yaml — verified identical for all 9 arms.
# Both halves now come from the single source: the name from label_for(), the style group from
# paper_scorecard_group() (its alias table lives in the yaml, not here).
_PS_ARMS = {f"vjepa_2_1_{_enc}": paper_scorecard_group(_enc)
            for _a, _enc, _grp, _kind in display_arms(include_heads=False)}
_PS_ARMS["vjepa_2_1_frozen"] = paper_scorecard_group("frozen")
# head-only arms duplicate the pretrain ENCODER on encoder-side metrics → excluded
# (they would draw 3 identical bars and confuse the family story); cassle/ewc are null.
# 2026-07-22: legends name the FAMILY only. They used to restate member arms inline
# ("(3-stage-DI, no-DI)", "(raw, auto-RGN)", "continual-SSL") in wording that no longer matched the
# bar labels those same arms carry — the same stale-duplication that let one arm print six different
# names. The bars are individually labelled from arm_registry, so a family legend must not re-name
# them. Also dropped the em-dashes and the "iter18" internal jargon from a paper-facing figure.
_PS_GROUP_STYLE = {
    "ours_flagship":    dict(color="#1B5E20", edge="black", lw=1.8, label="Surgery (ours): flagship arms"),
    "surgery_ablation": dict(color="#66BB6A", edge="#2E7D32", lw=0.8, label="Surgery (ours): ablation arms"),
    "ft_baseline":      dict(color="#90A4AE", edge="#546E7A", lw=0.8, label="Prior adaptation baselines"),
    "frozen":           dict(color="#E53935", edge="#B71C1C", lw=1.2, label="Frozen V-JEPA (baseline to beat)"),
    "improvement":      dict(color="#00897B", edge="#00695C", lw=1.4, label="Surgery (ours): improvement arms"),
}
# (the old _PS_GRP2PS alias dict + the "fill in any arm missing from the literal" loop that stood here
#  are both gone — _PS_ARMS above is now built from the registry for EVERY scheduler arm in one pass,
#  heads excluded as before since they duplicate the pretrain encoder on encoder-side metrics.)
# (json-key, pretty title, direction)  — higher='hi', lower='lo'
_PS_HERO = [
    ("causal", "HERO  ·  Causal L1   (sensitivity to causal perturbation)", "lo"),
    ("fut",    "HERO  ·  Future-frame MSE   (world-model prediction)",      "lo"),
]
_PS_SUPPORT = [
    ("mcos",      "Motion-cosine   (motion coherence)", "hi"),
    ("maskratio", "Mask-ratio slope   (masking robustness)", "lo"),
]


def _ps_bb_key(enc_2b):
    """Re-prefix a 2B-champion _PS_ARMS key (vjepa_2_1_<arm_enc>) to the CURRENT backbone's encoder name
    via _mw_enc_name — so the paper scorecard matches the 1B (vjepa_2_1_vitg_*) and 2.0 (vjepa_2_0_vitg_*)
    json keys, not only the 2B champion. 2B champion → identity (unchanged). 2026-06-24: _PS_ARMS keys were
    hardcoded to the bare 2B prefix, so on the 1B backbone NONE matched the json → a BLANK paper scorecard
    (the '[warn] arms absent from json' listing every arm)."""
    return _mw_enc_name(enc_2b[len("vjepa_2_1_"):])


def _ps_load(metrics_json: Path):
    rows = {}
    keymap = {_ps_bb_key(e): e for e in _PS_ARMS}   # backbone-correct json key → the canonical 2B _PS_ARMS key
    for d in json.load(open(metrics_json)):
        enc = d["encoder"]
        if enc not in keymap:
            continue
        rows[keymap[enc]] = d            # store under the _PS_ARMS key so _ps_panel's iteration matches
    missing = [a for a in _PS_ARMS if a not in rows]
    if missing:
        print(f"  [warn] arms absent from json (skipped): {missing}")
    return rows


def _ps_panel(ax, rows, key, title, direction, annotate_family=False):
    hi = direction == "hi"
    items = []
    for enc, grp in _PS_ARMS.items():
        if enc not in rows:
            continue
        cell = rows[enc][key]
        if cell["mean"] is None:
            continue
        # name from the SINGLE source, so this panel agrees with the forest / scale / train figures
        items.append((label_for(enc[len("vjepa_2_1_"):]), grp, cell["mean"], cell["ci_half"] or 0.0))
    items.sort(key=lambda t: -t[2] if hi else t[2])     # best first (top of chart)
    if not items:                       # partial LIVE run: no encoder has this metric scored yet → blank panel
        ax.text(0.5, 0.5, "awaiting data", transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#999", fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=11.5, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        return
    ys = np.arange(len(items))[::-1]
    # iter18 (2026-06-13, user order): auto common-exponent so clustered small decimals
    # separate — rescale the value axis + bar labels and tag the exponent in the title.
    scale, exp = common_exponent([m for *_, m, _ in items], [ci for *_, ci in items])
    for y, (short, grp, m, ci) in zip(ys, items):
        st = _PS_GROUP_STYLE[grp]
        ax.barh(y, m * scale, color=st["color"], edgecolor=st["edge"], linewidth=st["lw"],
                height=0.7, xerr=ci * scale, capsize=2.5,
                error_kw=dict(lw=1.0, ecolor="#263238"))
        star = "* " if grp == "ours_flagship" else ""
        ax.text((m + ci) * scale, y, f"  {star}{fmt_mantissa(m * scale)}", va="center", fontsize=8.5,
                fontweight="bold" if grp == "ours_flagship" else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([s for s, *_ in items], fontsize=9)
    arrow = "↑ higher better" if hi else "↓ lower better"
    ax.set_title(f"{title}     ({arrow}){exp_axis_tag(exp)}", fontweight="bold", fontsize=11.5, loc="left")
    lo = min((m - ci) * scale for _, _, m, ci in items)
    hivv = max((m + ci) * scale for _, _, m, ci in items)
    pad = (hivv - lo) * 0.16 or abs(hivv) * 0.05
    ax.set_xlim(lo - pad, hivv + pad * 3.4)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    if annotate_family:
        # shade the contiguous top run of surgery-family bars
        fam = {"ours_flagship", "surgery_ablation"}
        top_run = 0
        for _, grp, _, _ in items:
            if grp in fam:
                top_run += 1
            else:
                break
        if top_run >= 2:
            ax.axhspan(ys[top_run - 1] - 0.45, ys[0] + 0.45,
                       color="#A5D6A7", alpha=0.22, zorder=0)

        # best-ours vs raw ablation tie
        def get(name_sub):
            for s, g, m, ci in items:
                if name_sub in s:
                    return m, ci
            return None
        ours = min((get(n) for n in ("3-stage-DI", "no-DI") if get(n)),
                   key=lambda t: (t[0] if not hi else -t[0]), default=None)
        raw = get("raw")
        tie = ours and raw and abs(ours[0] - raw[0]) <= ours[1] + raw[1]
        # ── ONE annotation in the GUARANTEED-EMPTY corner ──
        # ↓ metrics sort asc → shortest bars at TOP → top-right is whitespace.
        # ↑ metrics sort desc → shortest bars at BOTTOM → bottom-right is whitespace.
        lines = []
        if top_run >= 2:
            lines.append(f"SURGERY FAMILY — sweeps top {top_run}")
        if tie:
            lines.append("* DI/3-stage ≈ raw surgery (95% CI)\n   → schedule adds no sig. gain")
        if lines:
            cy, va = (0.955, "top") if not hi else (0.045, "bottom")
            ax.text(0.985, cy, "\n".join(lines), transform=ax.transAxes,
                    ha="right", va=va, fontsize=8.2, fontweight="bold",
                    color="#1B5E20", linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#F1F8E9",
                              ec="#7CB342", alpha=0.95))


def plot_paper_scorecard(metrics_json: Path, out_dir: Path, stem: str = "eval_scorecard_paper"):
    """HONEST paper scorecard — surgery-family predictive dominance over the FULL baseline set (nothing
    hidden). Reads eval_metrics.json (same file the watch refresh emits). Folded verbatim from
    scripts/iter18_paper_scorecard.py build() — figure kept byte-equivalent."""
    rows = _ps_load(metrics_json)
    fig = plt.figure(figsize=(20, 16))
    # dedicated rows: hero · support · legend · banner — no overlap by construction
    gs = fig.add_gridspec(4, 2, height_ratios=[1.25, 1.0, 0.10, 0.32],
                          hspace=0.40, wspace=0.34,
                          left=0.17, right=0.965, top=0.91, bottom=0.03)
    # hero row
    for j, (k, t, d) in enumerate(_PS_HERO):
        _ps_panel(fig.add_subplot(gs[0, j]), rows, k, t, d, annotate_family=True)
    # support row
    for j, (k, t, d) in enumerate(_PS_SUPPORT):
        _ps_panel(fig.add_subplot(gs[1, j]), rows, k, t, d, annotate_family=True)

    # legend — own row, won't collide with the banner
    handles = [Patch(facecolor=s["color"], edgecolor=s["edge"], label=s["label"])
               for s in _PS_GROUP_STYLE.values()]
    axl = fig.add_subplot(gs[2, :]); axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=4, fontsize=10.5, frameon=True)

    # narrative banner — own row
    ax = fig.add_subplot(gs[3, :]); ax.axis("off")
    banner = (
        "HONEST READ (full baseline set — nothing hidden):  the FACTOR-SURGERY FAMILY occupies the top of every predictive-quality "
        "metric — causal sensitivity and future-frame MSE — with 95% CIs clear of full fine-tune, LoRA/DoRA and the frozen encoder.\n"
        "The DI / 3-stage schedule (ours) MATCHES raw surgery within 95% CI: an honest ablation, not a hidden weakness — the gain "
        "comes from surgery itself.   Surgery trades a small temporal-ordering / TCC margin (see full 14-panel scorecard) for this "
        "predictive lead; we report that trade-off rather than dropping those panels."
    )
    ax.text(0.5, 0.5, banner, transform=ax.transAxes, ha="center", va="center",
            fontsize=11, wrap=True,
            bbox=dict(boxstyle="round,pad=0.6", fc="#FFFDE7", ec="#F9A825", lw=1.2))

    fig.suptitle("Factor surgery dominates predictive-quality metrics — full baseline set, no per-panel hiding "
                 "(* = ours · green band = surgery family · 95% BCa CI)",
                 fontsize=15, fontweight="bold", y=0.965)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"  [paper-scorecard] saved → {p}")
    plt.close(fig)


# ═══ Big eyeball-able TCC comparison chart (folded from scripts/iter18_tcc_chart.py, iter18 2026-06-14) ═══
# Two horizontal-bar panels (TCC Kendall τ ↑ · TCC cycle-back ↓). Every encoder with TCC data is shown; the 4
# named players (Frozen / Pretrain / OURS-3stage-DI / raw) are boldly coloured, the rest grey. PLAIN decimals
# (no ×10ⁿ rescale) so they verify by eye against eval_metrics.csv. _tcc_*-prefixed to avoid name collisions;
# figure kept byte-equivalent (direct fig.savefig png@150 + pdf).
# display name + colour + bold? — keyed by encoder. The 4 named players pop; rest grey.
# 2026-07-22: PALETTE-ONLY now — (colour, highlight). The per-arm label this used to carry ("Pretrain
# (start pt)", "OURS  (3-stage DI)", …) was the 6th competing label source; names come from label_for()
# in plot_tcc_chart. Colours stay local by the registry's own contract (per-figure palettes are not
# roster metadata); only the 4 highlighted players get a bold hue, everything else is grey.
_TCC_ARMS = {
    "vjepa_2_1_frozen":                     ("#E53935", True),
    "vjepa_2_1_pretrain_encoder":           ("#1565C0", True),
    "vjepa_2_1_surgical_3stage_DI_encoder": ("#1B5E20", True),
    "vjepa_2_1_surgery_raw_encoder":        ("#C0CA33", True),
    "vjepa_2_1_surgical_noDI_encoder":      ("#9E9E9E", False),
    "vjepa_2_1_surgical_autorgn_encoder":   ("#9E9E9E", False),
    "vjepa_2_1_full_ft_encoder":            ("#9E9E9E", False),
    "vjepa_2_1_lpft_encoder":               ("#9E9E9E", False),
    "vjepa_2_1_peft_lora_encoder":          ("#9E9E9E", False),
    "vjepa_2_1_peft_dora_encoder":          ("#9E9E9E", False),
}
# iter18 (2026-06-14): any scheduler encoder NOT explicitly styled above → grey default (same treatment as
# the other fine-tuners), so a NEW arm auto-appears in this chart with no edit here. Roster from the single
# source (configs/arm_registry.yaml); the 4 named players above keep their bold custom colours.
for _a, _enc, _grp, _kind in display_arms():
    _full = f"vjepa_2_1_{_enc}"
    if _full not in _TCC_ARMS:
        _TCC_ARMS[_full] = ("#9E9E9E", False)
_TCC_PANELS = [
    ("tcc_tau",   "TCC Kendall τ   (↑ higher = better)", "higher",
     "frame-order rank correlation"),
    ("tcc_cycle", "TCC cycle-back   (↓ lower = better)",      "lower",
     "cycle-consistency error"),
]


def _tcc_panel(ax, rows, key, title, direction, subtitle, frozen_val):
    hi = direction == "higher"
    # name from the SINGLE source (label_for); _TCC_ARMS supplies only the palette + highlight flag
    items = [(label_for(e[len("vjepa_2_1_"):]), _TCC_ARMS[e][0], _TCC_ARMS[e][1],
              d[key]["mean"], d[key]["ci_half"] or 0.0)
             for e, d in rows.items() if e in _TCC_ARMS and d[key]["mean"] is not None]
    items.sort(key=lambda t: -t[3] if hi else t[3])          # best at TOP
    ys = np.arange(len(items))[::-1]
    for y, (name, col, bold, m, ci) in zip(ys, items):
        ax.barh(y, m, color=col, alpha=0.95 if bold else 0.5, height=0.66,
                edgecolor="black" if bold else col, linewidth=1.8 if bold else 0.6,
                xerr=ci, capsize=4, error_kw=dict(lw=1.2, ecolor="#222"))
        ax.text(m + ci, y, f"  {m:.3f}", va="center", fontsize=12.5,
                fontweight="bold" if bold else "normal",
                color="#000" if bold else "#666")
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, *_ in items], fontsize=12.5)
    for tick, (_, col, bold, *_2) in zip(ax.get_yticklabels(), items):
        tick.set_color(col if bold else "#777")
        tick.set_fontweight("bold" if bold else "normal")
    # "the bar to beat" — Frozen reference line
    ax.axvline(frozen_val, color="#E53935", ls="--", lw=1.6, alpha=0.7, zorder=0)
    ax.text(frozen_val, len(items) - 0.3, " Frozen = the bar to beat",
            color="#E53935", fontsize=10.5, fontweight="bold", va="bottom")
    lo = min(m - ci for *_3, m, ci in items)
    hivv = max(m + ci for *_3, m, ci in items)
    pad = (hivv - lo) * 0.18
    ax.set_xlim(lo - pad, hivv + pad * 2.6)                   # zoom so gaps are visible
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel(f"score  ({subtitle}) — exact decimals, = eval_metrics.csv", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


def plot_tcc_chart(metrics_json: Path, out_dir: Path, stem: str = "tcc_comparison"):
    """Big eyeball-able TCC-only comparison chart (Kendall τ + cycle-back). Reads eval_metrics.json. Folded
    verbatim from scripts/iter18_tcc_chart.py build() — figure kept byte-equivalent. Requires the frozen +
    3stage-DI + raw rows (FAIL LOUD via KeyError if the json is missing them — same as the original)."""
    rows = {x["encoder"]: x for x in json.load(open(metrics_json))}
    frozen = rows["vjepa_2_1_frozen"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
    for ax, (key, title, direction, sub) in zip(axes, _TCC_PANELS):
        _tcc_panel(ax, rows, key, title, direction, sub, frozen[key]["mean"])
    o = rows["vjepa_2_1_surgical_3stage_DI_encoder"]
    take = (f"Frozen wins BOTH · every fine-tuner drops below it · "
            f"OURS (τ {o['tcc_tau']['mean']:.3f}) is the GENTLEST surgery — "
            f"beats raw ({rows['vjepa_2_1_surgery_raw_encoder']['tcc_tau']['mean']:.3f}) "
            f"but still below Frozen ({frozen['tcc_tau']['mean']:.3f}).")
    fig.suptitle("TCC — can the encoder keep video frames in time-order?   "
                 "(bold = the 4 players · grey = other fine-tuners · whiskers = 95% CI)",
                 fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.015, take, ha="center", fontsize=12.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9C4", ec="#F9A825", lw=1.2))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.13, wspace=0.45)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"  [tcc-chart] saved → {p}")
    plt.close(fig)


def _winner_row(ax, tiesets, ny, mode):
    """Render the hero's bottom WINNER row in one of 3 honest styles and RETURN its height in grid
    units (so the caller extends ylim). tiesets[j] = {"sorted":[(e,v,ci) best-first], "tie":[e...
    95%-CI co-leaders], "leader":e, "lead_is_surg":bool}. "OURS" = the 4 GREEN surgery-novelty arms
    only (_OURS_GREEN) — argn (Auto-RGN baseline) and raw (surgery-on-raw control) are NOT credited
    as OURS even though their names contain 'surg' (the figure already colours them orange/brown).
    No style claims a significance the CIs don't support — ties are always marked; OURS (green) is
    highlighted only where it genuinely co-leads.
      · coleader_set     — stack ALL co-leaders, each in its OWN arm-colour (OURS green+bold); '+k more' if >4
      · surgery_coleader — name the best OURS co-leader (green); '~' unless it's the sole #1; else the true leader + 'OURS #rank'
      · tie_badge        — '~ tie (n)' when ≥2 co-leaders, else '<arm> WIN (sole)'; OURS tints the cell green"""
    ncol = max(tiesets, default=-1) + 1

    def _ours(e):
        return e.replace("vjepa_2_1_", "") in _OURS_GREEN

    def _empty(j):
        ax.add_patch(Rectangle((j - 0.5, ny - 0.5), 1, 1, facecolor=(0.85, 0.85, 0.85, 0.35),
                               edgecolor="white", lw=1.0, zorder=2))
        ax.text(j, ny, "—", ha="center", va="center", fontsize=10, color="black", zorder=3)

    if mode == "coleader_set":
        nmax = max((len(ts["tie"]) for ts in tiesets.values()), default=1)
        shown = min(nmax, 4)
        dy, over = 0.30, (nmax > 4)
        row_h = 0.5 + (shown + (1 if over else 0)) * dy + 0.1
        for j in range(ncol):
            ax.add_patch(Rectangle((j - 0.5, ny - 0.5), 1, row_h, facecolor=(0.96, 0.96, 0.96, 0.7),
                                   edgecolor="white", lw=1.0, zorder=2))
            ts = tiesets.get(j)
            if not ts:
                _empty(j)
                continue
            for k, e in enumerate(ts["tie"][:4]):         # each co-leader in its OWN arm colour; OURS bold
                ax.text(j, ny - 0.20 + k * dy, _short_label(e), ha="center", va="center", fontsize=6.5,
                        fontweight="bold" if _ours(e) else "normal", color=_color_for(e, j), zorder=3)
            if len(ts["tie"]) > 4:
                ax.text(j, ny - 0.20 + 4 * dy, f"+{len(ts['tie']) - 4} more", ha="center", va="center",
                        fontsize=5.5, color="#999999", style="italic", zorder=3)
        return row_h

    if mode == "tie_badge":
        for j in range(ncol):
            ts = tiesets.get(j)
            if not ts:
                _empty(j)
                continue
            n = len(ts["tie"])
            sole_ours = (n == 1 and _ours(ts["leader"]))
            any_ours = any(_ours(e) for e in ts["tie"])
            face = (_SURG_GREEN if sole_ours else (0.80, 0.90, 0.80, 0.55) if any_ours
                    else (0.85, 0.85, 0.85, 0.45))
            ax.add_patch(Rectangle((j - 0.5, ny - 0.5), 1, 1, facecolor=face,
                                   edgecolor="white", lw=1.0, zorder=2))
            if n == 1:
                ax.text(j, ny - 0.16, _short_label(ts["leader"]), ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white" if sole_ours else "black", zorder=3)
                ax.text(j, ny + 0.24, "WIN (sole)", ha="center", va="center", fontsize=6, color="black", zorder=3)
            else:
                ax.text(j, ny - 0.14, "~ tie", ha="center", va="center", fontsize=8, fontweight="bold",
                        color="black", zorder=3)
                ax.text(j, ny + 0.24, f"n={n}", ha="center", va="center", fontsize=7, color="black", zorder=3)
        return 1.0

    # default: surgery_coleader
    for j in range(ncol):
        ts = tiesets.get(j)
        if not ts:
            _empty(j)
            continue
        ours = next((e for (e, _v, _c) in ts["sorted"] if _ours(e)), None)
        if ours is None or ours not in ts["tie"]:        # no OURS arm co-leads → name the true leader, honestly
            le = ts["leader"]
            ax.add_patch(Rectangle((j - 0.5, ny - 0.5), 1, 1, facecolor=_color_for(le, j), alpha=0.5,
                                   edgecolor="white", lw=1.0, zorder=2))
            ax.text(j, ny - 0.16, _short_label(le), ha="center", va="center", fontsize=7,
                    fontweight="bold", color="black", zorder=3)
            if ours is not None:
                rank = [e for (e, _v, _c) in ts["sorted"]].index(ours) + 1
                ax.text(j, ny + 0.24, f"OURS #{rank}", ha="center", va="center", fontsize=6,
                        color=_SURG_GREEN, zorder=3)
            continue
        sole = (ts["tie"] == [ours])                     # OURS is the ONLY co-leader → a clear win
        ax.add_patch(Rectangle((j - 0.5, ny - 0.5), 1, 1, facecolor=_SURG_GREEN, alpha=0.55,
                               edgecolor="white", lw=1.0, zorder=2))
        ax.text(j, ny - 0.16, ("" if sole else "~") + _short_label(ours), ha="center", va="center",
                fontsize=7, fontweight="bold", color="black", zorder=3)
        ax.text(j, ny + 0.24, "clear #1" if sole else f"tie:{len(ts['tie'])}", ha="center",
                va="center", fontsize=6, color="black", zorder=3)
    return 1.0


def plot_hero_heatmap(metrics: dict, encoders: list, frozen: str, output_dir: Path):
    """B2: per-metric RAW-VALUE heatmap WITH numbers. TOP row = FROZEN baseline, then the contender arms,
    cols = hero metrics. Each cell PRINTS that arm's raw metric value 95% BCa CI [min, max] in native
    units — the paired surgery−pretrain comparison is NOT in the cells (it lives in the WINNER row + the
    dedicated m13_paired_diff_heatmap). Colour = good-oriented per-metric min-max WITH FROZEN INCLUDED, so
    a reader can read each arm's growth over the frozen baseline (green = best, red = worst), RdYlGn."""
    cat = [c for c in _hero_catalog(metrics) if c[3] in ("higher", "lower")]  # win/loss metrics only — signed (order) is diagnostic, lives in hero-table CSV
    contenders = [e for e in encoders if e != frozen]
    if not cat or not contenders:
        print("  [hero-heatmap] need a frozen baseline + ≥1 contender + metrics — skip")
        return
    rows = [frozen] + contenders                        # FROZEN baseline = row 0 (top), then the arms
    raw = np.full((len(rows), len(cat)), np.nan)        # RAW metric value (drives per-metric colour)
    cells = {}                                          # (i,j) -> (raw_lo, raw_hi) — the value's 95% CI
    for i, e in enumerate(rows):
        for j, (key, _fam, _on, direction, _yl, _cap, _lay) in enumerate(cat):
            be = metrics[key]["by_encoder"].get(e)
            if be is None:
                continue
            val, ciw = be                               # raw value + symmetric BCa half-width
            raw[i, j] = val
            cells[(i, j)] = (val - ciw, val + ciw)      # the value's 95% CI [min, max], native units
    norm = np.full_like(raw, np.nan)                    # colour = good-oriented per-metric min-max, FROZEN included → shows growth
    for j, (_k, _f, _o, direction, *_z) in enumerate(cat):
        col = raw[:, j]
        fin = col[np.isfinite(col)]
        if fin.size == 0:
            continue
        lo_, hi_ = float(fin.min()), float(fin.max())
        t = (col - lo_) / (hi_ - lo_) if hi_ > lo_ else col * 0.0 + 0.5  # 0..1, high raw = 1
        if direction == "lower":
            t = 1.0 - t                                 # low value = good = green
        norm[:, j] = 2.0 * t - 1.0                      # 0..1 → -1..1 for RdYlGn vmin/vmax = -1/+1
    fig, ax = plt.subplots(figsize=(3.0 + 1.75 * len(cat), 2.6 + 1.2 * (len(rows) + 1)))
    # alpha=0.6 → pastel RdYlGn (matches the hero TABLE); keeps red→green meaning while staying
    # light enough for plain BLACK BOLD numbers — no halo/outline needed (iter16 fix).
    im = ax.imshow(np.ma.masked_invalid(norm), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto", alpha=0.6)
    ax.grid(False)                                      # iter16 fix: kill the distracting white gridlines
    ax.tick_params(length=0)
    ax.set_xticks(range(len(cat)))
    ax.set_xticklabels([f"{c[0]}\n{_DIR_TAG[c[3]]}" for c in cat], rotation=45, ha="right", fontsize=9)
    _WTICK = {"coleader_set": "CO-LEADERS\n(95%CI #1)", "surgery_coleader": "WINNER\n(surgery)",
              "tie_badge": "STAT #1\n(95%CI)"}
    ax.set_yticks(list(range(len(rows))) + [len(rows)])
    ax.set_yticklabels([f"{_display_label(frozen)}  (baseline)"]
                       + [_display_label(e) for e in contenders]
                       + [_WTICK.get(_WINNER_MODE, "WINNER")], fontsize=10)
    # iter18 2026-06-08 (user order): the bottom row reports the 95%-CI CO-LEADER set per metric —
    # EVERY arm statistically tied for #1 (its CI overlaps the point-best) — NOT a single point-best.
    # At POC the CIs are wide enough that a SURGERY arm co-leads EVERY metric, so the old single-name
    # row under-sold it. The blue SPOTLIGHT now boxes EVERY co-leader cell (surgery boxed on each
    # metric it ties). Three honest render styles (env HERO_WINNER_MODE) — none invents significance.
    tiesets = {}     # j -> {"sorted":[(e,v,ci) best-first], "tie":[e... CI-tied #1], "leader":e, "lead_is_surg":bool}
    for j, (key, _fam, _on, direction, *_z) in enumerate(cat):
        cands = [(e, metrics[key]["by_encoder"][e][0], metrics[key]["by_encoder"][e][1] or 0.0)
                 for e in contenders
                 if e in metrics[key]["by_encoder"]
                 and metrics[key]["by_encoder"][e][0] is not None
                 and np.isfinite(metrics[key]["by_encoder"][e][0])]
        if not cands:
            continue
        cands.sort(key=lambda x: x[1], reverse=(direction != "lower"))   # best raw value first
        le, lv, lci = cands[0]
        tie = [e for (e, v, ci) in cands if abs(v - lv) <= (ci + lci)]    # 95% CI overlaps the point-leader
        tiesets[j] = {"sorted": cands, "tie": tie, "leader": le,
                      "lead_is_surg": _arm_family(le) == "surgery"}
    _ri = {e: i for i, e in enumerate(rows)}             # display-row index (frozen=0, arms 1..n)
    spotlight = {(_ri[e], j) for j, ts in tiesets.items() for e in ts["tie"] if e in _ri}
    for (i, j), (rlo, rhi) in cells.items():
        win = (i, j) in spotlight
        # Each cell shows ONLY that arm's RAW metric value 95% CI [min, max].
        _ci = "nan" if (np.isnan(rlo) or np.isnan(rhi)) else f"[{_fmt_val(rlo)},\n{_fmt_val(rhi)}]"
        ax.text(j, i, _ci, ha="center", va="center",
                fontsize=11 if win else 9, fontweight="bold", color="black")
        if win:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#1a3cff", linewidth=3.0, zorder=4))
    ny = len(rows)
    row_h = _winner_row(ax, tiesets, ny, _WINNER_MODE)   # render the bottom row per mode → its height
    ax.set_ylim(ny - 0.5 + row_h, -0.5)                  # extend bottom to include the WINNER row(s)
    ax.axhline(0.5, color="#444444", lw=1.3, ls="--")    # separator BELOW the FROZEN baseline (row 0)
    ax.axhline(ny - 0.5, color="black", lw=1.5)           # separator above the WINNER row
    _TITLE_TAIL = {
        "coleader_set": "BOTTOM = CO-LEADER SET: every arm whose 95% CI overlaps the point-best "
                        "(statistically tied for #1), surgery in green · BLUE BOX = each co-leader cell",
        "surgery_coleader": "BOTTOM = best SURGERY arm among the 95%-CI co-leaders (green; ~ = it's a "
                            "statistical tie, not a significant win) · BLUE BOX = every co-leader cell",
        "tie_badge": "BOTTOM = statistical status: '~ tie (n)' when the top arms' 95% CIs overlap, "
                     "else 'WIN (sole)' · BLUE BOX = every co-leader cell (surgery boxed where it ties)"}
    ax.set_title("HERO — raw metric value  ·  per cell: that arm's value 95% BCa CI [min, max] · "
                 "TOP ROW = FROZEN baseline (read each arm's growth over it)\ncolour: red = worst, "
                 "green = best (per-metric min-max incl. frozen, good-oriented) · "
                 + _TITLE_TAIL.get(_WINNER_MODE, _TITLE_TAIL["coleader_set"]),
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="red = worst   →   green = best (per-metric normalized value, good-oriented)")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_hero_raw_values"))
    print(f"  [hero-heatmap] m13_hero_raw_values.{{png,pdf}} — {len(rows)} rows (frozen + {len(contenders)} arms) × {len(cat)} (raw value CI)")


def plot_frozen_scorecard(metrics: dict, frozen_encoders: list, output_dir: Path, boot_str: str):
    """FROZEN-only leaderboard — ABSOLUTE head-metric values (NO Δ) for the image/video baselines +
    the V-JEPA frozen reference. rows = encoders (sorted best→worst on the 1st metric), cols = the
    metrics ALL frozen encoders share (the 3 head metrics; predictor metrics are V-JEPA-only → not
    comparable across baselines, excluded). Per-metric colour (green = best), best cell blue-boxed.
    Kept separate from the trained-arm hero views, which these baselines would clutter with N/A."""
    cat = _hero_catalog(metrics)
    cols = [c for c in cat if c[3] in ("higher", "lower")
            and all(metrics[c[0]]["by_encoder"].get(e) is not None for e in frozen_encoders)]
    if not cols or len(frozen_encoders) < _MIN_COMPARABLE:
        print(f"  [frozen-scorecard] skip — need ≥2 frozen encoders + ≥1 shared metric "
              f"(got {len(frozen_encoders)} enc, {len(cols)} shared metrics)")
        return
    cmap = plt.get_cmap("RdYlGn")
    colstat, winner = {}, {}
    for (key, _f, _o, direction, _y, _c, _l) in cols:
        vals = {e: metrics[key]["by_encoder"][e][0] for e in frozen_encoders}
        colstat[key] = (min(vals.values()), max(vals.values()))
        winner[key] = (max if direction == "higher" else min)(vals, key=vals.get)
    k0, d0 = cols[0][0], cols[0][3]                               # sort encoders by the 1st metric (good-first)
    ordered = sorted(frozen_encoders, key=lambda e: metrics[k0]["by_encoder"][e][0],
                     reverse=(d0 == "higher"))
    text_rows, colour_rows, win_coords = [], [], []
    csv_rows = [["encoder"] + [c[0] for c in cols]]
    for ri, e in enumerate(ordered):
        cells, ccols = [], []
        for ci, (key, _f, _o, direction, _y, _c, _l) in enumerate(cols):
            val, ciw = metrics[key]["by_encoder"][e]
            cells.append(f"{val:.3f}\n±{ciw:.3f}")
            vmin, vmax = colstat[key]
            t = 0.5 if vmax == vmin else (val - vmin) / (vmax - vmin)
            if direction == "lower":
                t = 1.0 - t                                      # low value = good = green
            r, g, b, _ = cmap(t)
            ccols.append((r, g, b, 0.55))
            if winner[key] == e:
                win_coords.append((ri, ci))
        text_rows.append(cells); colour_rows.append(ccols)
        csv_rows.append([e] + cells)
    row_labels = [_display_label(e) for e in ordered]
    col_labels = [f"{c[0]}\n{_DIR_TAG[c[3]]}" for c in cols]
    fig, ax = plt.subplots(figsize=(3.0 + 2.1 * len(cols), 1.6 + 0.6 * len(ordered)))
    ax.axis("off")
    tbl = ax.table(cellText=text_rows, cellColours=colour_rows,
                   rowLabels=row_labels, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 2.2)
    for (ri, ci) in win_coords:                                  # spotlight per-metric best: bold + blue box
        cell = tbl[(ri + 1, ci)]
        cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#1a3cff"); cell.set_linewidth(3.0)
    ax.set_title("FROZEN encoders — ABSOLUTE head-metric values (no Δ) · value ± BCa 95% CI · "
                 "colour = per-metric min-max (green = best) · BLUE BOX = best per metric\n"
                 f"{len(ordered)} frozen encoders × {len(cols)} shared metrics · sorted by {k0} · {boot_str}",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_frozen_scorecard"))
    import csv
    with open(str(output_dir / artifact("m13_frozen_scorecard")), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"  [frozen-scorecard] m13_frozen_scorecard.{{png,pdf,csv}} — {len(ordered)} frozen × {len(cols)} metrics")


# ── §3.3d SURGERY-vs-PRETRAIN verdict views — make "who wins overall" unmistakable ──
# The hero heatmap colours per-column (per-metric winners) but never aggregates pretrain-vs-
# surgery. These 3 views answer it directly, ALL off ONE canonical tally (_family_verdict =
# best-surgery vs best-pretrain CHAMPION DUEL per metric, with explicit TIES) so their headline
# numbers are IDENTICAL — resolving the earlier discrepancy (the scoreboard's per-arm argmax had
# spuriously awarded dead-heat metrics to pretrain via alphabetical tie-breaking).

_PREDICTOR_KEYS = {"future_mse", "rollout", "causal", "tdist", "teacher_free", "maskratio", "order"}
_TIE_EPS = 1e-3   # champion-duel |Δ| below this = dead heat. Values separate cleanly here
                  # (real wins ≥ 0.009, ties ≤ 6e-5) — nothing sits near the threshold.
_SHORT_DESC = {   # ≤8-word plain-English blurb per metric (under the x-axis; no blank space)
    "action_top1":  "classify motion class\nfrom frozen features",
    "motion_cos":   "same-motion clips cluster\nvs different motions",
    "taxonomy_f1":  "read scene attributes\nfrom features",
    "future_mse":   "predictor guesses\nmasked next-frame tokens",
    "rollout":      "error growth over\nmulti-step rollout",
    "causal":       "predict 2nd half\nfrom 1st half",
    "tdist":        "accuracy decay vs\nprediction horizon",
    "teacher_free": "free-run minus\nteacher-forced gap",
    "maskratio":    "degradation under\nheavier masking",
    "order":        "reliance on\nframe order",
}


def _arm_family(enc: str) -> str:
    if "frozen" in enc:
        return "frozen"
    if "surgical" in enc or "surgery" in enc:
        return "surgery"
    if "pretrain" in enc:
        return "pretrain"
    # iter18 2026-06-08: FT-technique baselines (B1-B4: Full-FT / LP-FT / PEFT-LoRA-DoRA / CaSSLe /
    # EWC) are the ABLATION COMPARISON SET — they must join the hero `core` (table + raw-value
    # heatmap), not the external-baseline scorecard. Without this they classified as "other" and
    # were silently dropped from m13_hero_raw_values / m13_hero_table.
    if any(t in enc for t in ("full_ft", "lpft", "peft", "cassle", "ewc")):
        return "ft"
    return "other"


def _pick_frozen_ref(encoders):
    """Frozen REFERENCE = the same-backbone frozen baseline of the trained arms, NOT just the
    first alphabetical 'frozen'. Once cross-arch image/video baselines (e.g. lejepa_vitL_frozen)
    land in the shared roots, the old `next(e for e in sorted(encoders) if 'frozen' in e)` mis-won
    the sort (lejepa < vjepa). Derive the arms' backbone stem (common prefix of surgery+pretrain
    arms) and prefer '{stem}_frozen'; fall back to the alphabetical-first 'frozen' when no trained
    arms are present (pure-baseline render)."""
    frozens = sorted(e for e in encoders if "frozen" in e)
    if not frozens:
        return None
    arms = sorted(e for e in encoders if _arm_family(e) in ("surgery", "pretrain", "ft"))
    if arms:
        lo, hi = arms[0], arms[-1]                          # common prefix of the arm names
        i = 0
        while i < len(lo) and i < len(hi) and lo[i] == hi[i]:
            i += 1
        cand = f"{lo[:i].rstrip('_')}_frozen"               # e.g. 'vjepa_2_1_frozen' (or '..._vitg_frozen')
        if cand in encoders:
            return cand
    return frozens[0]


def _backbone_of(enc):
    """Map a RAW encoder key → its trained-backbone id (vjepa_2_1_vitG / vjepa_2_1_vitg /
    vjepa_2_0_vitg) via _canon (champion's legacy vjepa_2_1_<arm> → vjepa_2_1_vitG_) + the _BB_TAG
    prefixes — the SAME single source of backbone identity the display labels use. External baselines
    (dinov2 / lejepa / any non-_BB_TAG key) → None (not one of the 3 trained backbones)."""
    c = _canon(enc)
    for pre, _long, _short in _BB_TAG:
        if c.startswith(pre):
            return pre.rstrip("_")
    return None


def _backbones_present(encoders) -> list:
    """Distinct trained backbones with ≥1 surgery/pretrain arm, in _BB_TAG display order
    (ViT-G → ViT-g → ViT-g·2.0). Drives the per-backbone hero breakouts."""
    have = {_backbone_of(e) for e in encoders if _arm_family(e) in ("surgery", "pretrain", "ft")}
    return [pre.rstrip("_") for pre, _l, _s in _BB_TAG if pre.rstrip("_") in have]


def _good_orient(d, lo, hi, direction):
    """Orient (Δ, lo, hi) so POSITIVE = better: lower-better → negate + swap bounds; else as-is."""
    if direction == "lower":
        return -d, -hi, -lo
    return d, lo, hi


def _direction_of(key: str) -> str:
    return next(c[3] for c in _CATALOG if c[0] == key)


def _good_vs_frozen_ci(metrics, key, arm, frozen):
    """Good-oriented Δ(arm − frozen) as (g, glo, ghi) (positive = better), or None."""
    dv = _delta_v_vs_frozen(metrics[key]["deltas"], arm, frozen)
    return None if dv is None else _good_orient(dv[0], dv[1], dv[2], _direction_of(key))


def _good_vs_frozen(metrics, key, arm, frozen):
    g = _good_vs_frozen_ci(metrics, key, arm, frozen)
    return None if g is None else g[0]


def _family_champion(metrics, key, frozen, arms):
    """Best arm in `arms` by good-Δ-vs-frozen for this metric (None if none scorable)."""
    scored = [(a, _good_vs_frozen(metrics, key, a, frozen)) for a in arms]
    scored = [(a, g) for a, g in scored if g is not None]
    return max(scored, key=lambda t: t[1])[0] if scored else None


def _scorable_keys(metrics):
    """Hero metrics minus 'signed' (order's sign is reliance, not a win/loss)."""
    return [k for k, *_ in _hero_catalog(metrics) if _direction_of(k) != "signed"]


def _wrap_name(enc: str) -> str:
    """Full display name wrapped onto 2 lines (y-axis labels — readable, space-saving)."""
    parts = _display_label(enc).split(" ")
    if len(parts) <= 1:
        return parts[0]
    mid = (len(parts) + 1) // 2
    return " ".join(parts[:mid]) + "\n" + " ".join(parts[mid:])


def _family_verdict(metrics, encoders, frozen):
    """CANONICAL surgery-vs-pretrain tally via champion duel (best surgery arm vs best pretrain
    arm) per metric, with explicit ties. Returns (n_surg, n_pre, n_tie,
    per{key→'surgery'|'pretrain'|'tie'}, arm_wins{arm→#outright metric wins}). Shared by all 3
    views so the headline is identical."""
    surg = [e for e in encoders if _arm_family(e) == "surgery"]
    pre = [e for e in encoders if _arm_family(e) == "pretrain"]
    ns = npr = nt = 0
    per, arm_wins = {}, {a: 0 for a in surg + pre}
    for key in _scorable_keys(metrics):
        sc, pc = _family_champion(metrics, key, frozen, surg), _family_champion(metrics, key, frozen, pre)
        if sc is None or pc is None:
            continue
        dv = _delta_v_vs_frozen(metrics[key]["deltas"], sc, pc)
        if dv is None:
            continue
        gd, glo, ghi = _good_orient(dv[0], dv[1], dv[2], _direction_of(key))
        # TIE = the PAIRED surgery−pretrain 95% BCa CI overlaps 0 → the two arms are statistically
        # indistinguishable on the SAME clips (the publishable test, NOT a mean point-estimate). A
        # degenerate (nan) CI falls back to the mean ε-guard. (iter17: was mean-only |gd|<_TIE_EPS.)
        is_tie = (abs(gd) < _TIE_EPS) if (np.isnan(glo) or np.isnan(ghi)) else (glo <= 0 <= ghi)
        if is_tie:
            per[key] = "tie"; nt += 1
        elif gd > 0:
            per[key] = "surgery"; ns += 1; arm_wins[sc] += 1
        else:
            per[key] = "pretrain"; npr += 1; arm_wins[pc] += 1
    return ns, npr, nt, per, arm_wins


def plot_scoreboard(metrics, encoders, frozen, output_dir):
    """#2 scoreboard: per-arm count of OUTRIGHT metric wins (champion duel; ties are NOT awarded
    to any arm — fixes the old alphabetical-tie-break that inflated pretrain) + the canonical
    SURGERY · PRETRAIN · TIE family banner shared with the other two views."""
    arms = [e for e in encoders if _arm_family(e) in ("surgery", "pretrain")]
    if not arms:
        print("  [scoreboard] no surgery/pretrain arms — skip"); return
    ns, npr, nt, _per, arm_wins = _family_verdict(metrics, encoders, frozen)
    order = sorted(arms, key=lambda a: arm_wins.get(a, 0))
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.55 * len(order) + 2)))
    y = np.arange(len(order))
    colors = [_color_for(a, i) for i, a in enumerate(order)]   # per-encoder colours (OURS green)
    ax.barh(y, [arm_wins.get(a, 0) for a in order], color=colors, alpha=0.85)
    for i, a in enumerate(order):
        ax.text(arm_wins.get(a, 0), i, f" {arm_wins.get(a, 0)}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([_wrap_name(a) for a in order], fontsize=8, rotation=20, va="center")
    ax.set_xlabel(f"# metrics won OUTRIGHT  (of {ns + npr + nt}; {nt} ties not awarded)")
    # name the duel's two sides with the SAME arm label the bars carry (arm_registry), so the title
    # cannot drift from the y-axis: this said "VANILLA CONT-SSL" while the bar read "pretrain".
    _pre = label_for("pretrain_encoder").upper()
    champ = "SURGERY" if ns > npr else (_pre if npr > ns else "TIE")
    fig.suptitle(f"SCOREBOARD —  SURGERY {ns} · {_pre} {npr} · TIE {nt}   (champion duel · winner: {champ})",
                 fontsize=13, fontweight="bold")
    ax.set_title(f"green = surgery arm · blue = {label_for('pretrain_encoder')} arm · "
                 "bar = outright metric wins by that arm", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, str(output_dir / "m13_scoreboard_surgery_vs_pretrain"))
    print(f"  [scoreboard] m13_scoreboard_surgery_vs_pretrain — surgery {ns} pretrain {npr} tie {nt}")


def plot_grouped_winner(metrics, encoders, frozen, output_dir):
    """#3 grouped heatmap: rows = PRETRAIN block then SURGERY block; cell = good-Δ-vs-frozen (bold)
    + its 95% CI on 2 lines; column-normalized colour; + SCORE column (mean per-metric-normalized,
    0–1) + WINNER row (canonical champion-duel: S/P/= ). Full wrapped y-names; ≤8-word metric
    blurbs under the x-axis. Block separators + verdict in title."""
    keys = _scorable_keys(metrics)
    pre = [e for e in encoders if _arm_family(e) == "pretrain"]
    surg = [e for e in encoders if _arm_family(e) == "surgery"]
    arms = pre + surg
    if not keys or not pre or not surg:
        print("  [grouped] need metrics + pretrain + surgery — skip"); return
    M = np.full((len(arms), len(keys)), np.nan)
    CI = {}
    for i, a in enumerate(arms):
        for j, k in enumerate(keys):
            gci = _good_vs_frozen_ci(metrics, k, a, frozen)
            if gci is not None:
                M[i, j] = gci[0]; CI[(i, j)] = (gci[1], gci[2])
    Mn = np.full_like(M, np.nan)
    for j in range(len(keys)):
        col = M[:, j]; lo, hi = np.nanmin(col), np.nanmax(col)
        Mn[:, j] = 0.5 if hi == lo else (col - lo) / (hi - lo)
    score = np.nanmean(Mn, axis=1)
    _ns, _np, _nt, per, _aw = _family_verdict(metrics, encoders, frozen)
    # iter18 H3: strict — _family_verdict emits a verdict for every key; an
    # unknown verdict value must KeyError (a 4th category should crash, not
    # silently render '·').
    winner = [{"surgery": "S", "pretrain": "V", "tie": "="}[per[k]] for k in keys]
    nrow, ncol = len(arms) + 1, len(keys) + 1
    rgba = np.ones((nrow, ncol, 4)); cmap = plt.cm.RdYlGn
    for j in range(len(keys)):
        for i in range(len(arms)):
            if not np.isnan(Mn[i, j]):
                rgba[i, j] = cmap(Mn[i, j]); rgba[i, j, 3] = 0.6
    slo, shi = np.nanmin(score), np.nanmax(score)
    for i, s in enumerate(score):
        t = 0.5 if shi == slo else (s - slo) / (shi - slo)
        rgba[i, ncol - 1] = cmap(t); rgba[i, ncol - 1, 3] = 0.6
    for j, w in enumerate(winner):
        rgba[nrow - 1, j] = (0.17, 0.63, 0.17, 0.6) if w == "S" else (
            (0.84, 0.15, 0.16, 0.6) if w == "V" else (0.6, 0.6, 0.6, 0.35))
    fig, ax = plt.subplots(figsize=(max(11.0, 1.05 * ncol + 3), 0.95 * nrow + 3))
    ax.imshow(rgba, aspect="auto")
    # Spotlight (blue box + bigger Δ): a DECISIVE metric → its single winning arm; a TIE → BOTH family
    # champions (best surgery cell + best pretrain cell) — a tie shows TWO boxes, not one.
    spotlight = set()
    for j, k in enumerate(keys):
        if not np.any(~np.isnan(M[:, j])):
            continue
        if per.get(k) == "tie":
            for champ in (_family_champion(metrics, k, frozen, surg), _family_champion(metrics, k, frozen, pre)):
                if champ in arms:
                    spotlight.add((arms.index(champ), j))
        else:
            spotlight.add((int(np.nanargmax(M[:, j])), j))
    for i in range(len(arms)):
        for j in range(len(keys)):
            if not np.isnan(M[i, j]):
                win = (i, j) in spotlight              # decisive→winner; tie→both champions
                ax.text(j, i - 0.24, _fmt_compact(M[i, j]), ha="center", va="center",
                        fontsize=15 if win else 10, fontweight="bold", color="black")
                if (i, j) in CI:
                    lo, hi = CI[(i, j)]
                    ax.text(j, i + 0.2, f"[{_fmt_compact(lo)},\n{_fmt_compact(hi)}]", ha="center", va="center",
                            fontsize=6, color="black")
                if win:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#1a3cff", linewidth=3.0, zorder=4))
        ax.text(ncol - 1, i, f"{score[i]:.2f}", ha="center", va="center", fontsize=10,
                fontweight="bold", color="black")
    for j, w in enumerate(winner):
        ax.text(j, nrow - 1, w, ha="center", va="center", fontsize=14, fontweight="bold", color="black")
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"{k}  {_DIR_TAG[_direction_of(k)]}\n{_SHORT_DESC.get(k, '')}" for k in keys]
                       + ["SCORE\n(norm 0–1)"],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([_wrap_name(a) for a in arms] + ["WINNER"], fontsize=8, rotation=20, va="center")
    ax.axhline(len(pre) - 0.5, color="black", lw=2)
    ax.axhline(len(arms) - 0.5, color="black", lw=1)
    ax.axvline(len(keys) - 0.5, color="black", lw=1)
    verdict = "SURGERY" if _ns > _np else ("VANILLA CONT-SSL" if _np > _ns else "SPLIT")
    ax.set_title(f"Δ vs frozen (good-oriented, value + 95% CI) · VANILLA-CONT-SSL block / SURGERY block · "
                 f"SCORE = mean per-metric-normalized (0–1) · BLUE BOX + big Δ = best arm per metric\n"
                 f"WINNER row (champion duel — S=surgery V=vanilla-cont-SSL ==tie · tie = paired 95% CI overlaps 0):  "
                 f"surgery {_ns} · vanilla-cont-SSL {_np} · tie {_nt}  →  {verdict}", fontsize=10)
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_grouped_winner_surgery_vs_pretrain"))
    print(f"  [grouped] m13_grouped_winner_surgery_vs_pretrain — surgery {_ns} pretrain {_np} tie {_nt}")


def plot_all_metric_bars(metrics: dict, output_dir: Path, boot_str: str) -> int:
    """One bar-with-CI panel per available catalog metric → <output_dir>/{head,predictor,encoder}/.
    Returns the number of panels written."""
    n = 0
    for key, family, out_name, direction, ylabel, caption, layman in _CATALOG:
        if direction == "signed":
            print(f"  [skip] {key}: signed (no better/worse direction) — diagnostic only, kept in CSV not plotted")
            continue
        md = metrics.get(key)
        if md is None:
            print(f"  [skip] {key}: source JSON absent")
            continue
        be, na = md["by_encoder"], md["na"]
        encoders = sorted(set(be) | na)
        if not encoders:
            print(f"  [skip] {key}: no encoders")
            continue
        vals = [be.get(e, (0.0, 0.0))[0] for e in encoders]
        errs = [be.get(e, (0.0, 0.0))[1] for e in encoders]
        sort_dir = "asc" if direction == "lower" else "desc"
        s_enc, s_vals, s_errs = _sort_by_metric(encoders, vals, errs, na, direction=sort_dir)
        order_word = "↑ lowest" if direction == "lower" else "↓ highest"
        badge = "" if direction == "signed" else direction
        sub = output_dir / _FAMILY_DIR[family]
        sub.mkdir(parents=True, exist_ok=True)
        _emit_bar(sub / out_name, s_enc, s_vals, s_errs, na, ylabel,
                  f"{key} — sorted {order_word} first", badge, caption, layman,
                  len(encoders), boot_str)
        n += 1
    return n


def plot_paired_forest(metrics, encoders, frozen, output_dir, boot_str):
    """Paired-difference FOREST plot: one row per metric = the champion-duel surgery−pretrain Δ + 95%
    BCa CI, on a per-metric SE-standardized x-axis so all metrics share ONE axis. Vertical line at 0 =
    no difference; shaded band = ±1.96 SE (the 95% 'not-significant' zone). A CI crossing 0 = TIE
    (grey); entirely right of 0 = surgery sig. better (green); entirely left = pretrain sig. better
    (red). Raw Δ [lo,hi] printed per row. Shows the DIFFERENCE directly (the convention for 'is A
    different from B across metrics' — forest plots; cf. Demšar 2006 JMLR critical-difference diagrams)
    so a tie is self-evident — no asking the reader to subtract two vs-frozen numbers."""
    surg = [e for e in encoders if _arm_family(e) == "surgery"]
    pre = [e for e in encoders if _arm_family(e) == "pretrain"]
    keys = _scorable_keys(metrics)
    if not surg or not pre or not keys:
        print("  [forest] need surgery + pretrain + metrics — skip")
        return
    rows = []
    for k in keys:
        sc = _family_champion(metrics, k, frozen, surg)
        pc = _family_champion(metrics, k, frozen, pre)
        if sc is None or pc is None:
            continue
        dv = _delta_v_vs_frozen(metrics[k]["deltas"], sc, pc)
        if dv is None:
            continue
        d, lo, hi = _good_orient(dv[0], dv[1], dv[2], _direction_of(k))
        rows.append((k, d, lo, hi))
    if not rows:
        print("  [forest] no paired deltas — skip")
        return
    clip, n = 5.0, len(rows)
    fig, ax = plt.subplots(figsize=(12.5, 0.62 * n + 2.6))
    ax.axvspan(-1.96, 1.96, color="#c8c8c8", alpha=0.30, zorder=0)        # 95% 'not significant' zone
    ax.axvline(0, color="black", lw=1.3, zorder=1)
    ns = npr = nt = 0
    for idx, (k, d, lo, hi) in enumerate(rows):
        if np.isnan(lo) or np.isnan(hi) or hi <= lo:                      # degenerate CI → mean-tie fallback
            tie = abs(d) < _TIE_EPS
            z = 0.0 if tie else (clip if d > 0 else -clip)
            zlo = zhi = z
        else:
            se = (hi - lo) / 3.9199                                       # 95% CI width = 2 × 1.96 SE
            z, zlo, zhi = d / se, lo / se, hi / se
            tie = lo <= 0 <= hi
        if tie:
            verdict, color = "tie", "#9e9e9e"; nt += 1
        elif d > 0:
            verdict, color = "SURGERY", "#2ca02c"; ns += 1
        else:
            verdict, color = "vanilla cont-SSL", "#d62728"; npr += 1
        zc = max(-clip, min(clip, z))
        zloc = max(-clip, min(clip, zlo))
        zhic = max(-clip, min(clip, zhi))
        ax.errorbar([zc], [idx], xerr=[[max(0.0, zc - zloc)], [max(0.0, zhic - zc)]], fmt="o",
                    color=color, ecolor=color, elinewidth=2.6, capsize=4, markersize=8, zorder=3)
        _ci_txt = "(≈ equal, no spread)" if (np.isnan(lo) or np.isnan(hi)) else f"[{_fmt_fine(lo)}, {_fmt_fine(hi)}]"
        ax.text(clip + 0.4, idx, f"{_fmt_fine(d)}  {_ci_txt}   {verdict}",
                ha="left", va="center", fontsize=8, fontweight="bold", color=color)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{r[0]}  {_DIR_TAG[_direction_of(r[0])]}" for r in rows], fontsize=9)
    ax.invert_yaxis()                                                     # first metric at the top
    ax.set_xlim(-clip - 0.6, clip + 6.8)
    ax.set_xticks([-clip, -1.96, 0, 1.96, clip])
    ax.set_xticklabels(["≤−5", "−1.96", "0", "+1.96", "≥+5"], fontsize=8)
    ax.set_xlabel("standardized surgery − vanilla cont-SSL effect (Δ / SE) · shaded = ±1.96 (95% not-significant) · 0 = no difference",
                  fontsize=9)
    _bb = _backbone_of(surg[0]) if surg else None
    _label = _BB_LABEL.get(_bb, _display_label(frozen))
    _v = "SURGERY" if ns > npr else ("VANILLA CONT-SSL" if npr > ns else "SPLIT")
    ax.set_title(f"PAIRED forest — surgery − vanilla cont-SSL (champion duel) · {_label}\n"
                 f"surgery {ns} · vanilla cont-SSL {npr} · tie {nt}  →  {_v}   (green = surgery sig · "
                 f"red = vanilla cont-SSL sig · grey = tie, CI crosses 0) · {boot_str}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_paired_forest_surgery_vs_pretrain"))
    print(f"  [forest] m13_paired_forest_surgery_vs_pretrain.{{png,pdf}} — {n} metrics · "
          f"surgery {ns} pretrain {npr} tie {nt}")


def plot_paired_diff_heatmap(metrics, encoders, output_dir, boot_str):
    """COMBINED, appealing surgery−pretrain view: a metrics × backbones grid (like the hero heatmaps)
    where each cell is the champion-duel PAIRED Δ (surgery − pretrain, good-oriented, + = surgery) — the
    DIFFERENCE itself, not two vs-frozen numbers — coloured RdYlGn by the standardized effect (deep
    green = surgery sig. better, deep red = pretrain, pale grey = TIE / 95% CI crosses 0), with the Δ +
    95% CI + verdict printed per cell, and a per-backbone S/P/tie summary row. The 'effect-size +
    significance heatmap' convention: dense + colourful AND a tie is self-evident."""
    bbs = _backbones_present(encoders)
    keys = _scorable_keys(metrics)
    if not bbs or not keys:
        print("  [paired-heatmap] no backbones / metrics — skip")
        return
    nk, nb = len(keys), len(bbs)
    grid, tally, labels = {}, {j: [0, 0, 0] for j in range(nb)}, {}
    for j, bb in enumerate(bbs):
        be = [e for e in encoders if _backbone_of(e) == bb]
        frz = _pick_frozen_ref(be)
        surg = [e for e in be if _arm_family(e) == "surgery"]
        pre = [e for e in be if _arm_family(e) == "pretrain"]
        labels[j] = "\n".join(_BB_LABEL.get(bb, bb).split(" · ")[:3])
        for i, k in enumerate(keys):
            sc = _family_champion(metrics, k, frz, surg) if surg else None
            pc = _family_champion(metrics, k, frz, pre) if pre else None
            if sc is None or pc is None:
                continue
            dv = _delta_v_vs_frozen(metrics[k]["deltas"], sc, pc)
            if dv is None:
                continue
            d, lo, hi = _good_orient(dv[0], dv[1], dv[2], _direction_of(k))
            if np.isnan(lo) or np.isnan(hi):
                tie, z = abs(d) < _TIE_EPS, 0.0
            else:
                se = (hi - lo) / 3.9199
                z, tie = (d / se if se > 0 else 0.0), (lo <= 0 <= hi)
            grid[(i, j)] = (d, lo, hi, tie, z)
            tally[j][0 if (not tie and d > 0) else (1 if (not tie and d < 0) else 2)] += 1
    fig, ax = plt.subplots(figsize=(3.0 + 2.8 * nb, 2.2 + 0.98 * nk))
    cmap = plt.cm.RdYlGn
    for (i, j), (d, lo, hi, tie, z) in grid.items():
        yy = nk - 1 - i
        if tie:
            face = (0.90, 0.90, 0.87, 1.0)
        else:
            r, g, b, _a = cmap(0.5 + 0.5 * max(-1.0, min(1.0, z / 4.0)))
            face = (r, g, b, 0.80)
        ax.add_patch(Rectangle((j, yy), 1, 1, facecolor=face, edgecolor="white", lw=1.6))
        ci = "≈ equal" if (np.isnan(lo) or np.isnan(hi)) else f"[{_fmt_fine(lo)}, {_fmt_fine(hi)}]"
        ax.text(j + 0.5, yy + 0.66, _fmt_fine(d), ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(j + 0.5, yy + 0.40, ci, ha="center", va="center", fontsize=7)
        ax.text(j + 0.5, yy + 0.15, "TIE" if tie else ("SURGERY" if d > 0 else "VANILLA CONT-SSL"),
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="black")
    for j in range(nb):
        ns, npr, nt = tally[j]
        ax.add_patch(Rectangle((j, -1), 1, 1, facecolor=(0.96, 0.96, 0.96, 1.0), edgecolor="white", lw=1.6))
        ax.text(j + 0.5, -0.5, f"Surgery {ns}\nVanilla cont-SSL {npr}\ntie {nt}", ha="center", va="center",
                fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, nb)
    ax.set_ylim(-1, nk)
    ax.set_xticks([j + 0.5 for j in range(nb)])
    ax.set_xticklabels([labels[j] for j in range(nb)], fontsize=8, fontweight="bold")
    ax.set_yticks([nk - 1 - i + 0.5 for i in range(nk)] + [-0.5])
    # NO ↑/↓ "better" tag: the cell Δ is already good-oriented (surgery − pretrain), so + = surgery won,
    # − = pretrain won regardless of the raw metric's direction — the per-metric arrow would mislead here.
    ax.set_yticklabels([keys[i] for i in range(nk)] + ["WINS"], fontsize=9)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("PAIRED  surgery − vanilla cont-SSL  (champion duel) across backbones · per cell: Δ + 95% BCa CI\n"
                 "deep green = surgery sig. better · deep red = vanilla cont-SSL · pale grey = TIE (95% CI crosses 0) · "
                 + boot_str, fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_paired_diff_heatmap"))
    print(f"  [paired-heatmap] m13_paired_diff_heatmap.{{png,pdf}} — {nk} metrics × {nb} backbones")


def _emit_hero_suite(metrics, core, frozen, out_dir, boot_str):
    """The 4 surgery-vs-pretrain hero views for ONE encoder set, written into out_dir: hero-table +
    Δ-vs-frozen heatmap always; scoreboard + grouped-winner only when BOTH families are present.
    Called once for the COMBINED set (all backbones → eval/) and once per backbone (eval/<backbone>/),
    so combined and per-backbone share ONE code path (identical semantics, no drift). Caller
    guarantees `frozen` ∈ core and len(core) ≥ 2."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_hero_table(metrics, core, frozen, out_dir, boot_str)
    plot_hero_heatmap(metrics, core, frozen, out_dir)
    # §3.3d — surgery-vs-pretrain verdict views (only when both families are present)
    if any(_arm_family(e) == "surgery" for e in core) and any(_arm_family(e) == "pretrain" for e in core):
        plot_scoreboard(metrics, core, frozen, out_dir)
        plot_grouped_winner(metrics, core, frozen, out_dir)
        plot_paired_forest(metrics, core, frozen, out_dir, boot_str)


def _vstack_panels(output_dir, backbones, src_name, out_name):
    """COMBINED overview = the per-backbone <src_name>.png panels APPENDED VERTICALLY into one PNG with
    a labelled header band per backbone. A pure image concat — NO cross-backbone math (each panel keeps
    its own arms vs its OWN frozen). Panels stacked top→bottom in the given order."""
    panels = [(bb, output_dir / bb / f"{src_name}.png") for bb in backbones]
    panels = [(bb, Image.open(str(p)).convert("RGB")) for bb, p in panels if p.exists()]
    if not panels:
        print(f"  [stack] no per-backbone {src_name} panels present — skip")
        return
    w = max(im.width for _, im in panels)
    # Big LABELLED header band per panel so each heatmap's BACKBONE + model size is unmistakable.
    fpath = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    longest = max((_BB_LABEL.get(bb, bb) for bb, _ in panels), key=len)
    fsize = 110
    font = ImageFont.truetype(fpath, fsize)
    while fsize > _FONT_MIN_PT and font.getlength(longest) > w * 0.94:       # auto-shrink to fit the panel width
        fsize -= 4
        font = ImageFont.truetype(fpath, fsize)
    band_h = int(fsize * 1.8)
    total_h = sum(band_h + im.height for _, im in panels)
    canvas = Image.new("RGB", (w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for bb, im in panels:
        label = _BB_LABEL.get(bb, bb)
        draw.rectangle([0, y, w - 1, y + band_h - 1], fill=(18, 28, 64))         # dark navy banner
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((w - tw) // 2, y + (band_h - th) // 2 - bbox[1]), label, fill="white", font=font)
        y += band_h
        canvas.paste(im, (0, y))           # left-aligned; a narrower panel just keeps white margin
        y += im.height
    canvas.save(str(output_dir / f"{out_name}.png"))
    canvas.save(str(output_dir / f"{out_name}.pdf"))               # both .png & .pdf (CLAUDE.md plot rule)
    print(f"  [stack] {out_name}.{{png,pdf}} — vertical append of {len(panels)} LABELLED backbone panels "
          f"({', '.join(bb for bb, _ in panels)}) · NO averaging")


# ── CLI ──────────────────────────────────────────────────────────────

def _skip_encoders_from_env(encoders):
    """Encoder names to HIDE from the m13 plots — honors ITER18_SKIP_ARMS (the run's --skip-arms),
    matching by arm-suffix with the surgery↔surgical rename so 'surgery_noDI_head' hides the encoder
    'vjepa_2_1[_vitg]_surgical_noDI_head'. Empty when the env isn't set (plot everything). The §3
    finale already passes a skip-filtered --encoders list; this is the belt-and-suspenders for the
    live preview + direct m13 runs."""
    raw = os.environ.get("ITER18_SKIP_ARMS", "").strip()
    if not raw:
        return set()
    _norm = lambda s: s.replace("surgical", "surgery")           # noqa: E731 (one-liner rename)
    # .strip("\"'") per token: a direct-prompt paste of the runbook's ITER18_SKIP_ARMS=\"$SKIP\" watch
    # form bakes LITERAL quotes into the value → '"cassle_encoder' / 'surgery_noDI_head"' won't match.
    skip = {_norm(a.strip().strip("\"'")) for a in raw.split() if a.strip().strip("\"'")}
    _pres = ("vjepa_2_1_vitG_", "vjepa_2_1_vitg_", "vjepa_2_0_vitg_", "vjepa_2_1_vitL_", "vjepa_2_1_")
    out = set()
    for e in encoders:
        suf = next((e[len(p):] for p in _pres if e.startswith(p)), e)
        if _norm(suf) in skip:
            out.add(e)
    return out


# ═══ §2b construct-validity: convergent/discriminant nomological-net heatmap (iter19) ═══════
# The hardest benchmark bar — SHOW the metrics measure their named construct, not "N numbers of
# unknown meaning". Reuses the eval_metrics.json matrix (no GPU). The 3 construct families are the
# documented umbrellas (head/probe · predictor · encoder-temporal). `future_mse`/`fut` is the JEPA
# PREDICTOR's L1 → grouped with predictor (its _CATALOG 'HEAD' tag is the legacy eval-band it ships
# in, NOT its construct; mirrors the CLAUDE.md glossary). Directions come from _CATALOG (single
# source). The statistics live in utils/validity.py (rule 32: it takes families/dirs as args).
_VALIDITY_SHORT2LONG = {"act": "action_top1", "tax": "taxonomy_f1",
                        "mcos": "motion_cos", "fut": "future_mse"}  # eval_metrics.json key ↔ _CATALOG key
_VALIDITY_CONSTRUCT = {"HEAD": "head/probe", "PRED": "predictor", "ENC": "encoder-temporal"}
# Human metric names for FIGURE tick labels — the SINGLE SOURCE is configs/metric_names.json (read via
# utils.metric_names). The scorecard's _MW_EVAL_METRICS AND the forest/scale _XB_* lists below all derive
# their names from THIS dict, so a rename in the json propagates to every plot at once — no re-typed variants
# (the 2026-07-08 forest "Teacher-free gap" vs scorecard "free-running exposure-bias gap" split is impossible now).
_VALIDITY_PLAIN = _load_metric_names()   # {key: full display name} ← configs/metric_names.json (FAIL LOUD if absent)


def _mn_hilo(_k):   # json better-direction → the forest/scale "hi"/"lo" ('signed' order shown positive = "hi")
    return "lo" if _mn_dir(_k) == "lower" else "hi"


def _mn_word(_k):   # json better-direction → the scorecard "higher"/"lower" ('signed' order shown "higher")
    return "lower" if _mn_dir(_k) == "lower" else "higher"
_VALIDITY_FAM_ORDER = {"head/probe": 0, "predictor": 1, "encoder-temporal": 2}
# DESIRABILITY colormap (overrides the textbook red=+/blue=- sign-convention — labeled on the
# colorbar): BLUE = +1 (metrics agree → convergent validity, BEST) · WHITE = 0 (independent) ·
# RED = -1 (disagree, WORST). Blue↔red is colour-blind-SAFE (unlike green↔red); endpoints kept
# MODERATE so the bold-black cell numbers stay legible (builtin RdBu ends too dark). .claude/plotting.md.
_VALIDITY_CMAP = LinearSegmentedColormap.from_list("rd_wt_bu", ["#C0392B", "#FFFFFF", "#2C6FBB"])


def _validity_metric_spec():
    """Ordered [(short_key, construct_family, direction)] derived from _CATALOG (single source).

    `future_mse` moves HEAD→predictor (the _CATALOG 'HEAD' tag is its eval-band, not its construct
    — it IS the JEPA predictor's L1). Stable-sorted by construct family so the heatmap's family
    blocks are contiguous; within a family, the original _CATALOG order is preserved."""
    long2short = {v: k for k, v in _VALIDITY_SHORT2LONG.items()}
    spec = []
    for key, fam, _out, direction, *_rest in _CATALOG:
        short = long2short.get(key, key)
        construct = "predictor" if key == "future_mse" else _VALIDITY_CONSTRUCT[fam]
        spec.append((short, construct, direction))
    spec.sort(key=lambda t: _VALIDITY_FAM_ORDER[t[1]])   # stable → keeps _CATALOG order within family
    return spec


def plot_metric_validity(metrics_json, out_dir, n_perm, seed):
    """§2b convergent/discriminant heatmap from an eval_metrics.json (subjects × metrics).

    Writes m13_metric_validity.{png,pdf,csv}: the Spearman correlation of the metrics (oriented
    higher=better, signed `order` excluded), with the 3 construct-family blocks outlined and the
    within / between / gap / permutation-p headline. Reuses the existing matrix — pure CPU, no GPU.
    """
    metrics_json, out_dir = Path(metrics_json), Path(out_dir)
    rows = json.loads(metrics_json.read_text())
    spec = _validity_metric_spec()

    def _val(r, k):
        v = r.get(k)
        return v["mean"] if isinstance(v, dict) and v.get("mean") is not None else np.nan

    # metrics actually present (non-null somewhere) in this json
    present = [s for s in spec if any(np.isfinite(_val(r, s[0])) for r in rows)]
    keys = [s[0] for s in present]
    families = [s[1] for s in present]
    directions = [s[2] for s in present]
    fam_set = sorted(set(families))

    # subjects = encoder rows with ≥1 non-null metric in EACH construct family. Drops the all-null
    # arms (cassle/ewc) and the head-only / predictor-only partial arms whose duplicate-encoder
    # rows would inflate the correlation; pairwise-complete then handles any remaining gaps.
    subjects, dropped = [], []
    for r in rows:
        vals = {k: _val(r, k) for k in keys}
        covered = {families[i] for i, k in enumerate(keys) if np.isfinite(vals[k])}
        if set(fam_set).issubset(covered):
            subjects.append((r["encoder"], [vals[k] for k in keys]))
        else:
            dropped.append((r["encoder"], sorted(set(fam_set) - covered)))
    for enc, miss in dropped:
        print(f"  [metric-validity] drop {enc} (no data in: {', '.join(miss)})")
    # iter19 fix-2: drop by-construction DUPLICATE rows — an arm that reuses another's encoder yields
    # an identical metric vector → a fake-perfect correlation. The family-coverage drop above already
    # removes the *_head/empty rows; this explicit guard keeps it honest as the board grows (v2).
    _seen, _uniq = [], []
    for enc, vec in subjects:
        key = tuple(round(x, 6) if np.isfinite(x) else None for x in vec)
        if key in _seen:
            print(f"  [metric-validity] drop {enc} — by-construction duplicate metric vector of an earlier row")
            continue
        _seen.append(key)
        _uniq.append((enc, vec))
    subjects = _uniq
    if len(subjects) < 4:
        print(f"  [metric-validity] SKIP — only {len(subjects)} fully-covered subjects (need ≥4)")
        return

    enc_names = [e for e, _ in subjects]
    M = np.array([v for _, v in subjects], dtype=float)         # subjects × metrics
    corr, npair = pairwise_spearman(orient_higher_better(M, directions))
    summ = family_summary(corr, families, n_perm=n_perm, rng=np.random.default_rng(seed))

    # signed metrics (order) are all-NaN after orientation → drop them from the figure
    keep = [j for j, d in enumerate(directions) if d != "signed"]
    keys_k = [keys[j] for j in keep]
    fam_k = [families[j] for j in keep]
    corr_k = corr[np.ix_(keep, keep)]

    # Styling per .claude/plotting.md: ALL text BLACK + BOLD (never white-on-dark — fails on a
    # projector / B&W print), and cell numbers sized to fill ~70% of the cell width (drop the
    # leading zero → fewer glyphs → bigger font). Analysis-phase eyeball figure.
    init_style()
    n = len(keys_k)
    cell_in = 0.85                                   # inches per cell → big cells so numbers read large
    side = cell_in * n
    cell_pt = cell_in * 72.0
    num_fs = 0.32 * cell_pt                          # 4-char value ('-.91') fills ~70% of the cell width
    tick_fs = 0.22 * cell_pt

    def _annot(v):                                   # 2dp, drop the leading zero so glyphs are bigger
        s = f"{v:.2f}"
        if s.startswith("0."):
            return s[1:]                             # 0.91 → .91
        if s.startswith("-0."):
            return "-" + s[2:]                       # -0.91 → -.91
        return s

    fig, ax = plt.subplots(figsize=(side + 3.8, side + 3.4))
    im = ax.imshow(np.where(np.isfinite(corr_k), corr_k, 0.0), cmap=_VALIDITY_CMAP, vmin=-1, vmax=1)
    plain = [_VALIDITY_PLAIN[k] for k in keys_k]   # full metric names on the figure (CLAUDE.md glossary)
    ax.set_xticks(range(n))
    ax.set_xticklabels(plain, rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=tick_fs, fontweight="bold", color="black")
    ax.set_yticks(range(n))
    ax.set_yticklabels(plain, rotation=45, ha="right", va="center", rotation_mode="anchor",
                       fontsize=tick_fs, fontweight="bold", color="black")
    for i in range(n):
        for j in range(n):
            txt = "·" if (i == j or not np.isfinite(corr_k[i, j])) else _annot(corr_k[i, j])
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=num_fs, fontweight="bold", color="black")   # ALL annotations black + bold
    start = 0
    for fam in sorted(set(fam_k), key=fam_k.index):
        size = fam_k.count(fam)
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), size, size, fill=False,
                               edgecolor="black", lw=3.0))
        ax.text(start + size / 2.0 - 0.5, -0.62, fam, ha="center", va="bottom",
                fontsize=tick_fs + 1.0, fontweight="bold", color="black")
        start += size
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ   ·   blue = +1 agree (convergent, best)   ·   red = -1 disagree (worst)",
                   fontsize=tick_fs, fontweight="bold", color="black")
    cbar.ax.tick_params(labelsize=tick_fs * 0.85, colors="black")
    tri = npair[np.triu_indices_from(npair, 1)]
    n_lo = int(tri[tri > 0].min()) if np.any(tri > 0) else 0
    ax.set_title(
        f"§2b construct validity — convergent / discriminant   ·   "
        f"N = {len(enc_names)} subjects (pairwise n {n_lo}–{int(npair.max())})\n"
        f"within-family ρ = {summ['within']:.2f}   >>   between-family ρ = {summ['between']:.2f}"
        f"   ·   gap = {summ['gap']:.2f}   (permutation p = {summ['perm_p']:.3f})\n"
        f"signed `order` excluded (reported-not-ranked)",
        fontsize=tick_fs * 0.9, fontweight="bold", color="black", pad=cell_pt * 1.15)
    save_fig(fig, str(out_dir / "m13_metric_validity"))
    plt.close(fig)

    # CSV: the oriented Spearman matrix + the summary header (literal stem, matching save_fig usage)
    csv_path = out_dir / "m13_metric_validity.csv"
    lines = [f"# within={summ['within']:.4f} between={summ['between']:.4f} gap={summ['gap']:.4f} "
             f"perm_p={summ['perm_p']:.4f} N={len(enc_names)} n_perm={n_perm} seed={seed}",
             "metric,family," + ",".join(keys_k)]
    for i, k in enumerate(keys_k):
        cells = ["" if not np.isfinite(corr_k[i, j]) else f"{corr_k[i, j]:.4f}"
                 for j in range(len(keys_k))]
        lines.append(f"{k},{fam_k[i]}," + ",".join(cells))
    csv_path.write_text("\n".join(lines) + "\n")
    print(f"  [metric-validity] within ρ={summ['within']:.2f} ≫ between ρ={summ['between']:.2f} "
          f"(gap {summ['gap']:.2f}, perm p={summ['perm_p']:.3f}, N={len(enc_names)}) "
          f"→ {out_dir / 'm13_metric_validity'}.{{png,pdf,csv}}")
    return summ


def plot_criterion_validity(metrics_json, criterion_csv, metric_key, out_dir, n_boot, seed):
    """§2b criterion validity — does `metric_key` track an EXTERNAL capability? Reads criterion_csv
    (rows: encoder,ext_score — e.g. a frozen action-probe on a public SSv2/Kinetics subset), aligns by
    encoder, renders a rank-scatter (subject rank by the metric vs by the external score) + Spearman ρ
    with a bootstrap 95% CI → m13_criterion_<metric_key>.{png,pdf,csv}.

    SKIPS (no crash) when criterion_csv is absent — the external score needs a GPU probe, so in a normal
    metrics_watch refresh this is a documented no-op until criterion.csv exists (iter19 plan §2b)."""
    import csv as _csv
    metrics_json, out_dir, criterion_csv = Path(metrics_json), Path(out_dir), Path(criterion_csv)
    if not criterion_csv.exists():
        print(f"  [criterion] skip — no external criterion at {criterion_csv} "
              f"(needs a GPU action-probe → encoder,ext_score CSV; iter19 §2b)")
        return None
    spec = {s[0]: s for s in _validity_metric_spec()}
    if metric_key not in spec:
        sys.exit(f"plot_criterion_validity: unknown metric_key {metric_key!r} (not in _CATALOG)")
    direction = spec[metric_key][2]
    ext = {}
    with open(criterion_csv, newline="") as f:
        for r in _csv.DictReader(f):
            ext[r["encoder"]] = float(r["ext_score"])
    rows = json.loads(metrics_json.read_text())

    def _mval(r):
        v = r.get(metric_key)
        m = v["mean"] if isinstance(v, dict) and v.get("mean") is not None else np.nan
        return (-m if direction == "lower" else m)        # orient higher=better

    pairs = [(r["encoder"], _mval(r), ext[r["encoder"]])
             for r in rows if r["encoder"] in ext and np.isfinite(_mval(r))]
    if len(pairs) < 4:
        print(f"  [criterion] skip — only {len(pairs)} subjects shared between {metric_key} + criterion (need ≥4)")
        return None
    encs = [p[0] for p in pairs]
    mx = np.array([p[1] for p in pairs])
    cy = np.array([p[2] for p in pairs])
    res = criterion_rho(mx, cy, n_boot=n_boot, rng=np.random.default_rng(seed))
    rx = np.argsort(np.argsort(mx))                        # ordinal ranks for the scatter
    ry = np.argsort(np.argsort(cy))

    init_style()
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.scatter(rx, ry, s=70, color="#2C6FBB", edgecolor="black", zorder=3)
    for e, a, b in zip(encs, rx, ry):
        ax.text(a, b, " " + _short_label(e), fontsize=6.5, va="center", color="black")
    ax.set_xlabel(f"subject rank by {_VALIDITY_PLAIN[metric_key]} (↑=better)",
                  fontsize=10, fontweight="bold", color="black")
    ax.set_ylabel("subject rank by EXTERNAL gold-standard (↑=better)",
                  fontsize=10, fontweight="bold", color="black")
    ci = f"[{res['lo']:.2f}, {res['hi']:.2f}]" if np.isfinite(res["lo"]) else "[n/a]"
    # iter19 fix-6: title stays NEUTRAL — the label depends on the anchor: a same-construct anchor
    # (e.g. act vs an SSv2 action-probe) is CONVERGENT validity (weak); a predictor metric (fut/causal)
    # vs a DOWNSTREAM task is true CRITERION validity. The plan states which per anchor; don't overclaim.
    ax.set_title(f"§2b external-anchor validity — {_VALIDITY_PLAIN[metric_key]} vs external\n"
                 f"Spearman ρ = {res['rho']:.2f}  {ci}   (N = {res['n']} subjects)",
                 fontsize=9.5, fontweight="bold", color="black")
    ax.grid(True, alpha=0.3, zorder=0)
    save_fig(fig, str(out_dir / f"m13_criterion_{metric_key}"))
    plt.close(fig)
    (out_dir / f"m13_criterion_{metric_key}.csv").write_text(
        f"# metric={metric_key} rho={res['rho']:.4f} lo={res['lo']:.4f} hi={res['hi']:.4f} "
        f"N={res['n']} n_boot={n_boot} seed={seed}\nencoder,metric_oriented,ext_score\n"
        + "\n".join(f"{e},{m:.6f},{c:.6f}" for e, m, c in pairs) + "\n")
    print(f"  [criterion] {metric_key} vs external: ρ={res['rho']:.2f} {ci} (N={res['n']}) "
          f"→ {out_dir / f'm13_criterion_{metric_key}'}.{{png,pdf,csv}}")
    return res


# ═══ metrics_watch regeneration (iter18 2026-06-12, user order) ═══════════════════════════
# SELF-CONTAINED copy of scripts/iter18_poc_metrics.py's figure + data-dump generation, so the
# probe_plot/metrics_watch/<BACKBONE>/ artifacts (train_trajectories / kept_scorecard /
# eval_scorecard .png+.pdf + train_metrics/eval_metrics .json+.csv) stay reproducible from
# src/ ALONE after the iter18 scheduler scripts move to scripts/legacy/. No import from
# scripts/ — the needed DAG constants are duplicated below (they are FROZEN once iter18 ends;
# this copy is the durable record). Invoke:
#   python src/m13_eval_plot.py --POC --output-dir outputs/poc/probe_plot \
#       --outputs-root outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k \
#       --train-root   outputs/poc/vjepa_2_1_vitG_2B/train \
#       --metrics-watch-out outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k/probe_plot/metrics_watch \
#       --metrics-watch-only
# Backbone via ITER18_BACKBONE env, hidden arms via ITER18_SKIP_ARMS env (same contract as
# the watch scripts; the scheduler-log fallback is dropped — env is the durable source).
_MW_BACKBONE = os.environ.get("ITER18_BACKBONE", "vjepa_2_1_vitG")
# iter18 (2026-06-14): these 4 roster maps are now READ FROM THE SINGLE SOURCE (configs/arm_registry.yaml)
# via src/utils/arm_registry.py — NOT hardcoded — so a NEW arm (e.g. the wiseft-v2 sweep) auto-appears in
# the metrics_watch refresh with no edit here. arm2enc()/arm2dir() return the scheduler arms in registry
# order (byte-equal to the old ARM2ENC/ARM2DIR literals + the post-06-12 improvement & merge arms);
# _MW_TRAIN_ORDER preserves registry file order (display_arms order == the table/figure order); _MW_HEAD_ARMS
# is the surgery/pretrain HEAD kinds. Merge arms (wiseft) carry no probe_history → empty train blocks, and
# _mw_render_graphs styles every arm via _MW_TRAIN_STYLE.get(arm, _MW_DEF_TRAIN_STYLE) (grey fallback).
_MW_ARM2ENC = arm2enc()
_MW_ARM2DIR = arm2dir()
_MW_TRAIN_ORDER = [a for a, _e, _g, _k in display_arms(include_merge=True)]
_MW_HEAD_ARMS = {a for a, _e, _g, k in display_arms(include_merge=True)
                 if k in ("surgery_head", "pretrain_head")}
_MW_PT_FAMILIES = ["rollout", "causal", "tdist", "maskratio", "order", "teacher_free"]
_MW_ET_FLAT = ["aot", "tov", "pace"]
_MW_FRESH_LOG_S = 1800


def _mw_enc_name(arm_enc):
    """Champion 2B vitG drops its size tag; other backbones keep the full name (iter17 rule)."""
    prefix = "vjepa_2_1" if _MW_BACKBONE == "vjepa_2_1_vitG" else _MW_BACKBONE
    return f"{prefix}_{arm_enc}"


def _mw_jload(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _mw_jsonl(p: Path) -> list:
    try:
        text = p.read_text()
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                return rows     # record mid-write under a live run — show what's complete
    return rows


def _mw_skip_arms():
    """Arms hidden from the GRAPHS (data files keep everything) — ITER18_SKIP_ARMS env only."""
    env = os.environ.get("ITER18_SKIP_ARMS", "").strip()
    return {tok.strip().strip("\"'") for tok in env.split() if tok.strip().strip("\"'")}


def _mw_arm_train_val_n(mtag: str, arm: str):   # no root arg: this reads logs/, not the outputs tree
    """(n_train, n_val) parsed from the arm's newest scheduler log when logs/ still exists —
    graceful (None, None) once the iter18 logs are archived; the json then records null."""
    logs = sorted((q for q in Path("logs").glob(f"iter18_ngpu_{mtag}_train_{arm}_*.log")
                   if q.exists()), key=lambda q: q.stat().st_mtime)
    if not logs:
        return None, None
    try:
        head = logs[-1].read_text(errors="replace")[:16000]
    except OSError:
        return None, None
    m = re.search(r"train/val split: ([\d,]+) train / ([\d,]+) val", head)
    if m:
        return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
    m = re.search(r"Train clips: ([\d,]+) \| Val clips: ([\d,]+)", head)
    if m:
        return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
    tr = re.search(r"Loaded subset: ([\d,]+) clip keys from train_pool", head)
    va = (re.search(r"Loaded subset: ([\d,]+) clip keys from val_split", head)
          or re.search(r"Loaded val subset: ([\d,]+) clips", head))
    return (int(tr.group(1).replace(",", "")) if tr else None,
            int(va.group(1).replace(",", "")) if va else None)


def _mw_train_blocks(train_root: Path, mtag: str):
    """One block per arm: every probe checkpoint + the selector-replay verdicts (🎯/✋/·).

    `train_root` is the directory that DIRECTLY contains the per-arm m09 dirs, i.e.
    outputs/<mode>/<backbone>_<size>/train/. It is a separate CLI arg (--train-root) rather than
    being derived from --outputs-root: the eval artifacts live under <backbone>/eval/<split>/ and
    the train artifacts under <backbone>/train/, so ONE root cannot address both. It used to be
    read as `outputs_root / _MW_BACKBONE / <m09-dir>`, which matched the flat iter18 layout but
    resolves to nothing today — so every arm silently produced an empty block and each refresh
    truncated train_metrics.json (21,505 -> 3,615 B on POC-2B, 2026-07-22) and blanked
    train_trajectories."""
    blocks = []
    for arm in _MW_TRAIN_ORDER:
        d = train_root / _MW_ARM2DIR[arm]
        hist = _mw_jsonl(d / "probe_history.jsonl")
        summ = _mw_jload(d / artifact("training_summary"))
        logs = sorted((q for q in Path("logs").glob(f"iter18_ngpu_{mtag}_train_{arm}_*.log")
                       if q.exists()), key=lambda q: q.stat().st_mtime)
        log_fresh = bool(logs) and (time.time() - logs[-1].stat().st_mtime) < _MW_FRESH_LOG_S
        status = "✅" if summ else ("🔄" if (hist or log_fresh) else "⬚")
        n_tr, n_va = _mw_arm_train_val_n(mtag, arm)
        rows, kept_i = [], None
        best = float("inf")
        for i, r in enumerate(hist):
            if arm in _MW_HEAD_ARMS:
                verdict = "·"
            elif r.get("future_l1") is not None and r["future_l1"] < best:
                best, kept_i, verdict = r["future_l1"], i, "🎯"
            else:
                verdict = "✋"
            rows.append((r, verdict))
        head_best = ""
        if arm in _MW_HEAD_ARMS and summ and "best_val_loss" in summ:
            head_best = f"sel=head-vloss {summ['best_val_loss']:.3f}@ep{summ.get('best_epoch', '?')}"
        blocks.append({"arm": arm, "st": status, "n_tr": n_tr, "n_va": n_va,
                       "rows": rows, "kept_i": kept_i, "head_best": head_best})
    return blocks


def _mw_eval_rows(outputs_root: Path):
    """Per-encoder (mean, ci_half) per metric from the eval artifacts — the _raw payload the
    figures + json/csv consume (the watch script's terminal display strings are not files)."""
    encs = [_mw_enc_name("frozen")] + [_mw_enc_name(e) for e in _MW_ARM2ENC.values()]
    rows = []
    for enc in encs:
        act = _mw_jload(outputs_root / artifact("probe_action_dir") / enc / artifact("test_metrics"))
        tax = _mw_jload(outputs_root / artifact("probe_taxonomy_dir") / enc / artifact("test_metrics"))
        mc = _mw_jload(outputs_root / artifact("probe_motion_cos_dir") / enc / "intra_inter_ratio.json")
        fm = _mw_jload(outputs_root / artifact("probe_future_mse_dir") / enc / "aggregate_mse.json")
        pt = {fam: _mw_jload(outputs_root / artifact("predictor_temporal_dir") / enc / f"aggregate_{fam}.json")
              for fam in _MW_PT_FAMILIES}
        et = {fam: _mw_jload(outputs_root / artifact("encoder_temporal_dir") / enc / f"aggregate_{fam}.json")
              for fam in _MW_ET_FLAT}
        tcc = _mw_jload(outputs_root / artifact("encoder_temporal_dir") / enc / "aggregate_tcc.json") or {}
        tax_macro = None
        if tax and isinstance(tax.get("dims"), dict) and tax["dims"]:
            vals = [v["test_mean"] for v in tax["dims"].values() if "test_mean" in v]
            tax_macro = sum(vals) / len(vals) if vals else None
        n_test = next((src["n_test"] for src in (act, fm, mc) if src and "n_test" in src), None)

        def _half(d, ci_key):
            return d.get(ci_key, {}).get("ci_half") if (d and isinstance(d.get(ci_key), dict)) else None

        rows.append({
            "_enc_full": enc,
            "n_te": ("—" if n_test is None else str(n_test), ""),
            "_raw": {
                "act":  (act["top1_acc"] if act and "top1_acc" in act else None, _half(act, "top1_ci")),
                "tax":  (tax_macro, None),
                "mcos": (mc["score_mean"] if mc and "score_mean" in mc else None, _half(mc, "score_ci")),
                "fut":  (fm["mse_mean"] if fm and "mse_mean" in fm else None, _half(fm, "mse_ci")),
                **{fam: ((pt[fam]["mean"], _half(pt[fam], "ci")) if pt[fam] and "mean" in pt[fam] else (None, None))
                   for fam in _MW_PT_FAMILIES},
                **{fam: ((et[fam]["mean"], _half(et[fam], "ci")) if et[fam] and "mean" in et[fam] else (None, None))
                   for fam in _MW_ET_FLAT},
                **{key: ((tcc[sub]["mean"], _half(tcc.get(sub), "ci"))
                         if isinstance(tcc.get(sub), dict) and "mean" in tcc[sub] else (None, None))
                   for key, sub in (("tcc_cycle", "cycle_back"), ("tcc_tau", "kendalls_tau"))},
            },
        })
    return rows


# OURS = the surgery NOVELTY family from the SINGLE source (registry groups), so a new improvement arm
# auto-counts as OURS in the kept-scorecard verdict (a winning tccaux/intervene no longer mislabels as
# "OTHERS LEAD"). Mirrors scripts/iter18_poc_metrics.py _FAM_OURS exactly. include_merge=True so the
# wiseft merge arms (also ours_*/improvement group) count too.
_MW_OURS_GROUPS = {"ours_flagship", "ours_head", "improvement"}
_MW_FAM_OURS = {a for a, _e, _g, _k in display_arms(include_merge=True) if _g in _MW_OURS_GROUPS}
# 2026-07-22: this map is now STYLE-ONLY — (color, linestyle, linewidth). It used to carry a 4th
# element, a hand-written short label ("vCSSL", "s3DI-enc", …), which made it the 3rd of four
# competing label sources in this file. Labels now come from arm_registry.label_for() via
# _mw_style() below, so the scorecard/train legends print the SAME arm name as every other figure.
_MW_TRAIN_STYLE = {  # arm → (color, linestyle, linewidth); OURS = greens
    "pretrain_encoder":          ("#1565C0", "-",  1.8),
    "surgery_3stage_DI_encoder": ("#1B5E20", "-",  2.6),
    "surgery_noDI_encoder":      ("#43A047", "-",  2.6),
    "surgery_3stage_DI_head":    ("#81C784", ":",  1.8),
    "surgery_noDI_head":         ("#A5D6A7", ":",  1.8),
    "surgical_autorgn_encoder":  ("#E65100", "--", 1.4),
    "surgery_raw_encoder":       ("#827717", "-.", 1.8),
    "full_ft_encoder":           ("#F57C00", "--", 1.4),
    "lpft_encoder":              ("#8D6E63", "--", 1.4),
    "peft_lora_encoder":         ("#78909C", "--", 1.4),
    "peft_dora_encoder":         ("#546E7A", "--", 1.4),
    "cassle_encoder":            ("#9E9D24", "--", 1.4),
    "ewc_encoder":               ("#6D4C41", "--", 1.4),
    # iter18 (2026-06-14) improvement arms — GREENS (the recolor below forces them to the family green anyway);
    # any FUTURE/merge arm falls back to _MW_DEF_TRAIN_STYLE.
    "surgery_3stage_DI_replay25_encoder":  ("#2E7D32", "-", 2.2),
    "surgery_3stage_DI_diheavy_encoder":   ("#2E7D32", "-", 2.2),
    "surgery_3stage_DI_tccaux_encoder":    ("#2E7D32", "-", 2.2),
    "surgery_3stage_DI_intervene_encoder": ("#2E7D32", "-", 2.2),
}
_MW_DEF_TRAIN_STYLE = ("#BDBDBD", ":", 1.2)   # unregistered/merge arm → grey (a new arm never KeyErrors the graph)


def _mw_style(arm: str) -> tuple:
    """(label, color, linestyle, linewidth) for a train-graph arm. The label ALWAYS resolves through
    arm_registry (never the old literal, never the '?' placeholder a merge arm used to get), so a
    brand-new or merge arm shows its real name in grey rather than an unnamed line."""
    return (label_for(arm), *_MW_TRAIN_STYLE.get(arm, _MW_DEF_TRAIN_STYLE))
# user goal (2026-06-14): every variant of OUR SURGERY NOVELTY renders GREEN — never a non-green hue — so the
# surgery family is one visual cluster vs the distinctly-coloured baselines. "Our surgery" = registry kind ∈
# {surgery, surgery_head, merge} (the surgery trainers + the wiseft merges built FROM them); this INCLUDES
# surgery_raw (kind=surgery — it IS surgery on raw factors) but EXCLUDES surgical_autorgn (kind=baseline — a
# prior FT technique, NOT our novelty), which keeps its own colour. Shade MAY differ between sub-roles (the
# user allows "different of GREEN"), so the recolor keeps each arm's linestyle/width/label and forces only the
# COLOUR: the ours headline family (flagship + head + improvement, incl. wiseft merges) → ONE dark green;
# surgery_raw → a DISTINCT green so the ablation stays separable yet unmistakably surgery-green. kind-derived
# (single source), so a new surgery/improvement arm auto-greens with no edit here.
_MW_OURS_DARK_GREEN = "#2E7D32"
_MW_SURGERY_GREEN = {a: _MW_OURS_DARK_GREEN for a, _e, _g, k in display_arms(include_merge=True)
                     if k in ("surgery", "surgery_head", "merge")}
_MW_SURGERY_GREEN["surgery_raw_encoder"] = "#558B2F"   # ablation (still surgery) → distinct olive-green
_MW_TRAIN_STYLE = {a: ((_MW_SURGERY_GREEN[a], s[1], s[2]) if a in _MW_SURGERY_GREEN else s)
                   for a, s in _MW_TRAIN_STYLE.items()}   # 3-tuple now: colour is s[0], not s[1]
_MW_TRAIN_METRICS = [
    ("probe_top1",    "action top-1",  "higher"),
    ("motion_cos",    "motion-cos",    "higher"),
    ("future_l1",     "future L1",     "lower"),
    ("causal_l1",     "causal L1",     "lower"),
    ("maskratio",     "mask-ratio",    "lower"),
    ("val_jepa_loss", "val JEPA loss", "lower"),
]
# scorecard panels (key, NAME, direction) — EVERYTHING derived from configs/metric_names.json: keys+order
# from ordered_keys(), name from _VALIDITY_PLAIN, direction from the json ('order' is canonical 'signed' →
# shown "higher" per the scorecard's "uses-time = better" convention, via _mn_word). No metric literals here.
_MW_EVAL_METRICS = [(_k, _VALIDITY_PLAIN[_k], _mn_word(_k)) for _k in _mn_keys()]
_MW_EVAL_RAW_KEYS = ["act", "tax", "mcos", "fut", "rollout", "causal", "tdist", "maskratio",
                     "order", "teacher_free", "aot", "tov", "pace", "tcc_cycle", "tcc_tau"]
_MW_TRAIN_PROBE_KEYS = ["step", "probe_top1", "motion_cos", "future_l1", "causal_l1",
                        "maskratio", "val_jepa_loss"]


def _mw_render_graphs(blocks, ev, out_dir: Path):
    """The 3 metrics_watch figures (verbatim recipe of the watch script — uses this module's
    own _bar_with_ci/_sort_by_metric, which the watch script itself imported from here)."""
    def _kept_row(b):
        if b["kept_i"] is not None:
            return b["rows"][b["kept_i"]][0]
        return b["rows"][-1][0] if b["rows"] else None

    # ── F1: train trajectories ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    arms_drawn = []
    for ax, (key, title, direction) in zip(axes.flat, _MW_TRAIN_METRICS):
        panel_ys = []
        for b in blocks:
            sty = _mw_style(b["arm"])   # (label, colour, ls, lw) — merge/new arm → grey, no KeyError
            pts = sorted((r["step"], r[key]) for r, _ in b["rows"]
                         if r.get(key) is not None and r.get("step") is not None)
            if not pts:
                continue
            if b["arm"] not in arms_drawn:
                arms_drawn.append(b["arm"])
            xs, ys = zip(*pts)
            panel_ys.extend(ys)
            ax.plot(xs, ys, sty[2], color=sty[1], lw=sty[3], marker="o", ms=3.5)
            kr = _kept_row(b)
            if b["kept_i"] is not None and kr.get(key) is not None:
                ax.plot(kr["step"], kr[key], marker="*", ms=15, color=sty[1],
                        mec="black", mew=0.8, ls="none", zorder=5)
        # iter18 (2026-06-13): common-exponent on the y-axis (data untouched, ticks relabelled).
        scale, exp = common_exponent(panel_ys)
        if exp != 0:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, s=scale: fmt_mantissa(v * s)))
        arrow = "↑ better" if direction == "higher" else "↓ better"
        ax.set_title(f"{title}  ({arrow}){exp_axis_tag(exp)}", fontweight="bold", fontsize=13)
        ax.set_xlabel("global step", fontsize=10)
        ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], color=_mw_style(a)[1], ls=_mw_style(a)[2],
                          lw=_mw_style(a)[3], marker="o", ms=4, label=_mw_style(a)[0])
               for a in _MW_TRAIN_STYLE if a in arms_drawn]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7),
                   frameon=True, fontsize=11)
    fig.suptitle("iter18 TRAIN probe checkpoints — every arm, every probe · star marker = KEPT ckpt"
                 " · OURS = greens (solid enc / dotted head)", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.91, bottom=0.13, hspace=0.35, wspace=0.25)
    save_fig(fig, str(out_dir / "train_trajectories"))
    plt.close(fig)

    # ── F2: KEPT-checkpoint scorecard + verdict strip ──
    _CI_KEY = {"probe_top1": "top1_ci_half", "motion_cos": "motion_cos_ci_half",
               "future_l1": "future_l1_ci_half"}
    _Z95 = 1.96

    def _row_ci(row, key):
        ck = _CI_KEY.get(key)
        if ck and row.get(ck) is not None:
            return float(row[ck])
        if key == "probe_top1" and row.get("n_probe_clips"):
            p_, n_ = float(row["probe_top1"]), int(row["n_probe_clips"])
            return _Z95 * (p_ * (1 - p_) / n_) ** 0.5
        return None

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.38], hspace=0.5, wspace=0.45)
    axes6 = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    verdicts = []
    for ax, (key, title, direction) in zip(axes6, _MW_TRAIN_METRICS):
        entries = []
        for b in blocks:
            row = _kept_row(b)
            if row is not None and row.get(key) is not None:
                entries.append((b["arm"], float(row[key]), _row_ci(row, key)))
        if not entries:
            ax.axis("off")
            ax.set_title(f"{title} — no data yet", fontsize=12)
            verdicts.append((title, None, None, None, None, None))
            continue
        entries.sort(key=lambda t: t[1], reverse=(direction == "higher"))
        labels = [_mw_style(a)[0] + (" (hd)" if a.endswith("_head") else "")
                  for a, _, _ in entries]
        colors = [_mw_style(a)[1] for a, _, _ in entries]
        vals = [v for _, v, _ in entries]
        errs = [(e or 0.0) for _, _, e in entries]
        scale, exp = common_exponent(vals, errs)     # iter18 (2026-06-13): rescale value axis
        sv = [v * scale for v in vals]
        se = [e * scale for e in errs]
        ys = np.arange(len(entries))[::-1]
        bars = ax.barh(ys, sv, color=colors, alpha=0.9, height=0.65,
                       xerr=se, capsize=3, error_kw={"lw": 1.1, "ecolor": "#222"})
        for y, (arm, _, _), v_s, e_s, b_ in zip(ys, entries, sv, se, bars):
            if arm in _MW_FAM_OURS:
                b_.set_edgecolor("black")
                b_.set_linewidth(1.8)
            ax.text(v_s + e_s, y, f" {fmt_mantissa(v_s)}" + (f"±{fmt_mantissa(e_s)}" if e_s else ""),
                    va="center", fontsize=8)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9)
        lo = min(v - e for v, e in zip(sv, se))
        hi = max(v + e for v, e in zip(sv, se))
        pad = (hi - lo) * 0.15 or abs(hi) * 0.05 or 0.01
        ax.set_xlim(lo - pad, hi + pad * 4.0)
        arrow = "↑" if direction == "higher" else "↓"
        ax.set_title(f"{title} {arrow} · best: {labels[0]}{exp_axis_tag(exp)}", fontweight="bold", fontsize=12)
        ours = [(v, e) for a, v, e in entries if a in _MW_FAM_OURS]
        other = [(v, e) for a, v, e in entries if a not in _MW_FAM_OURS]
        if ours and other:
            pick = max if direction == "higher" else min
            (bo, eo), (bx, ex) = pick(ours), pick(other)
            win = bo > bx if direction == "higher" else bo < bx
            verdicts.append((title, win, bo, bx, eo, ex))
        else:
            verdicts.append((title, None, None, None, None, None))
    axv = fig.add_subplot(gs[2, :])
    axv.axis("off")
    n = len(verdicts)
    for i, (title, win, bo, bx, eo, ex) in enumerate(verdicts):
        x0 = i / n
        if win is None:
            col, txt = "#ECEFF1", f"{title}\n(awaiting data)"
        else:
            tie = (eo is not None and ex is not None and abs(bo - bx) <= (eo + ex))
            col = "#FFF9C4" if tie else ("#C8E6C9" if win else "#FFCDD2")
            tag = "≈ TIE (CIs overlap)" if tie else ("OURS LEAD" if win else "OTHERS LEAD")
            txt = f"{title}\nOURS {bo:.4f} · other {bx:.4f}\n{tag}"
        axv.add_patch(plt.Rectangle((x0 + 0.004, 0.08), 1 / n - 0.008, 0.84,
                                    facecolor=col, edgecolor="#555",
                                    transform=axv.transAxes))
        axv.text(x0 + 1 / (2 * n), 0.5, txt, transform=axv.transAxes,
                 ha="center", va="center", fontsize=10, fontweight="bold")
    axv.text(0.0, 1.06, "VERDICT — best OURS (surgery; black-edged bars) vs best OTHER · "
                        "whiskers = 95% CI (per-clip BCa; legacy top-1 rows: binomial approx; "
                        "causal/mask-ratio/val-loss have no per-clip data → no whisker)",
             transform=axv.transAxes, fontsize=11, fontweight="bold", va="bottom")
    fig.suptitle("iter18 KEPT-checkpoint scorecard — the encoder each arm EXPORTS to eval",
                 fontsize=15, fontweight="bold")
    save_fig(fig, str(out_dir / "kept_scorecard"))
    plt.close(fig)

    # ── F3: EVAL scorecard (15 panels on a 4×4 grid — incl. the signed 'order' diagnostic; 1 cell blank) ──
    have_any = any(r["_raw"][k][0] is not None for r in ev for k, _, _ in _MW_EVAL_METRICS)
    if not have_any:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "no EVAL artifacts yet —\nper-encoder evals launch as each arm finishes training",
                ha="center", va="center", fontsize=16, fontweight="bold", color="#666")
        save_fig(fig, str(out_dir / "eval_scorecard"))
        plt.close(fig)
        return
    fig, axes14 = plt.subplots(4, 4, figsize=(30, 22))
    for ax in axes14.flat[len(_MW_EVAL_METRICS):]:
        ax.axis("off")
    for ax, (k, title, direction) in zip(axes14.flat, _MW_EVAL_METRICS):
        # AUTO-DROP scoreless encoders (user order 2026-07-22). At FULL scale only 3 of the 24
        # registered arms are evaluated, so this used to draw 21 hatched placeholder bars whose
        # rotated x-labels overlapped into an unreadable smear across every panel. A bar with no
        # measurement carries no information — omit the encoder entirely rather than render "N/A".
        # Filtering is PER PANEL, so an arm missing only one metric still appears in the others;
        # each panel is complete for its own metric.
        rows_k = [r for r in ev if r["_raw"][k][0] is not None]
        if not rows_k:
            ax.text(0.5, 0.5, "awaiting data", transform=ax.transAxes, ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#999")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        encs = [r["_enc_full"] for r in rows_k]
        vals = [r["_raw"][k][0] for r in rows_k]
        errs = [(r["_raw"][k][1] or 0.0) for r in rows_k]
        s_enc, s_val, s_err = _sort_by_metric(encs, vals, errs, set(),
                                              "desc" if direction == "higher" else "asc")
        _bar_with_ci(ax, s_enc, s_val, s_err, ylabel=title,
                     title=title, na_set=None, direction=direction)
    # Banner heading (model identity) — mirrors the m13_grouped_winner stacked overview so the 1B and 2B
    # scorecards self-identify, especially once stacked into the combined PDF. _BB_LABEL = single-source caption.
    _hdr = _BB_LABEL.get(_MW_BACKBONE, _MW_BACKBONE)
    fig.text(0.5, 0.992, _hdr, ha="center", va="top", fontsize=21, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#121C40", edgecolor="none"))
    fig.text(0.5, 0.952, "iter18 EVAL scorecard — per-encoder TEST artifacts · 95% BCa CI where available",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#555")
    fig.subplots_adjust(hspace=0.6, wspace=0.3, top=0.93)
    save_fig(fig, str(out_dir / "eval_scorecard"))
    plt.close(fig)


def _mw_dump(blocks, ev, out_dir: Path):
    """train_metrics.{json,csv} + eval_metrics.{json,csv} — same schema as the watch script."""
    import csv
    train = [{"arm": b["arm"], "status": b["st"], "n_train": b["n_tr"], "n_val": b["n_va"],
              "head_best": b["head_best"], "kept_idx": b["kept_i"],
              "probes": [{**{k: r.get(k) for k in _MW_TRAIN_PROBE_KEYS},
                          "verdict": v, "kept": (i == b["kept_i"])}
                         for i, (r, v) in enumerate(b["rows"])]}
             for b in blocks]
    (out_dir / "train_metrics.json").write_text(json.dumps(train, indent=1))
    with open(out_dir / "train_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "ckpt", *_MW_TRAIN_PROBE_KEYS, "verdict", "kept"])
        for t in train:
            for i, pr in enumerate(t["probes"]):
                w.writerow([t["arm"], i + 1, *[pr.get(k) for k in _MW_TRAIN_PROBE_KEYS],
                            pr["verdict"], pr["kept"]])
    ev_json = [{"encoder": r["_enc_full"], "n_test": r["n_te"][0],
                **{k: {"mean": r["_raw"][k][0], "ci_half": r["_raw"][k][1]}
                   for k in _MW_EVAL_RAW_KEYS if k in r["_raw"]}}
               for r in ev]
    (out_dir / "eval_metrics.json").write_text(json.dumps(ev_json, indent=1))
    with open(out_dir / "eval_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "n_test", *[c for k in _MW_EVAL_RAW_KEYS for c in (k, k + "_ci")]])
        for r in ev:
            raw = r["_raw"]
            cells = [c for k in _MW_EVAL_RAW_KEYS for c in raw.get(k, (None, None))]
            w.writerow([r["_enc_full"], r["n_te"][0], *cells])


def combine_scorecards_pdf(src_pdfs, out_pdf):
    """Stack per-backbone eval_scorecard PDFs into ONE combined comparison PDF (vector, via pymupdf
    show_pdf_page) — a single-file 1B-vs-2B view mirroring the m13_grouped_winner stacked overview. Each
    source grid already carries its own _BB_LABEL banner (the per-backbone heading), so the stack self-labels
    with no extra drawing here. Missing inputs are skipped with a warning (graceful when one backbone's
    scorecard is not built yet). Returns the out path, or None if no input PDF exists."""
    import fitz   # pymupdf — vector PDF compositing (no rasterization, no extra pip dep)
    srcs = []
    for p in src_pdfs:
        p = Path(p)
        if p.exists():
            srcs.append((p, fitz.open(str(p))))
        else:
            print(f"  [combine] skip missing {p}")
    if not srcs:
        print("  [combine] no input scorecard PDFs found — nothing to combine")
        return None
    rects = [d[0].rect for _, d in srcs]
    width = max(r.width for r in rects)
    height = sum(r.height for r in rects)
    out = fitz.open()
    page = out.new_page(width=width, height=height)
    y = 0.0
    for (p, d), r in zip(srcs, rects):
        x0 = (width - r.width) / 2.0          # centre a narrower grid; equal widths just fill
        page.show_pdf_page(fitz.Rect(x0, y, x0 + r.width, y + r.height), d, 0)
        y += r.height
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_pdf.with_suffix(f".tmp{os.getpid()}.pdf")   # atomic write — never half a PDF on disk
    out.save(str(tmp))
    os.replace(str(tmp), str(out_pdf))
    out.close()
    for _, d in srcs:
        d.close()
    _pdf_to_png(out_pdf, out_pdf.with_suffix(".png"))       # user 2026-07-01: raster PNG twin beside the PDF
    print(f"  [combine] stacked {len(srcs)} scorecard(s) → {out_pdf} (+ .png)  ({width:.0f}x{height:.0f}pt)")
    return out_pdf


def _pdf_to_png(pdf_path, png_path, dpi=120):
    """Rasterize a (single, possibly tall) PDF page to PNG via pymupdf — the raster twin the user wants
    beside the vector combined scorecard. Atomic write; no extra pip dep (pymupdf already used above)."""
    import fitz
    doc = fitz.open(str(pdf_path))
    pix = doc[0].get_pixmap(dpi=dpi)
    png_path = Path(png_path)
    tmp = png_path.with_suffix(f".tmp{os.getpid()}.png")
    pix.save(str(tmp))
    os.replace(str(tmp), str(png_path))
    doc.close()


# ══ Cross-backbone report (forest + scale-replication) — user 2026-07-01 ══════════════════════════════
# Reads BOTH backbones' metrics_watch/eval_metrics.json keyed on the ARM SUFFIX (so the 2B vjepa_2_1_<arm>
# and 1B vjepa_2_1_vitg_<arm> align), and renders two comparison figures into the cross-backbone
# metrics_watch root beside eval_scorecard_combined. Colours reuse _color_for → every mark matches the
# combined scorecard: surgery novelty = dark green (#2E7D32), competitors keep their own hues. OURs vs
# COMPetitors is derived SINGLE-SOURCE from that colour (dark-green == our surgery novelty).
_XB_OURS_GREEN = "#2E7D32"
# forest rows (key, "hi"|"lo", NAME) — ALL derived from configs/metric_names.json: the 13 keys from
# forest_keys() (tax + order omitted there), direction via _mn_hilo, name from _VALIDITY_PLAIN. No literals,
# so every forest tick == the scorecard panel title ("free-running exposure-bias gap", never "Teacher-free gap").
_XB_METRICS = [(_k, _mn_hilo(_k), _VALIDITY_PLAIN[_k]) for _k in _mn_forest()]
# task2 2026-07-01: scale_replication covers ALL 15 metrics (the eval_metrics glossary), not just 4. The
# forest keeps its 13-row _XB_METRICS; this superset adds taxonomy-F1 + frame-order for the 5×3 scatter grid.
# scale-replication panels (key, "hi"|"lo", NAME) for ALL 15 — derived from configs/metric_names.json
# (ordered_keys + _mn_hilo + _VALIDITY_PLAIN). No metric literals; panels re-sort by ρ at render time.
_XB_ALL15 = [(_k, _mn_hilo(_k), _VALIDITY_PLAIN[_k]) for _k in _mn_keys()]


def _xb_is_ours(suffix):
    """OURs = the surgery-novelty dark-green arms PLUS surgery_raw (task1 2026-07-01: the surgery-on-raw
    ablation is counted WITH OURs, not the competitors)."""
    return suffix == "surgery_raw_encoder" or _color_for(f"vjepa_2_1_vitG_{suffix}", 0) == _XB_OURS_GREEN


def _xb_arm_short(suffix):
    """The ONE cross-figure arm display name — full variant name minus the '_encoder' suffix. Used by
    the forest winner column AND the scale-replication legend (VM14-class 2026-07-13: they had two
    different strippers, so the same arm read 'surgical_3stage_DI_diheavy' in one figure and
    '3stage_DI_diheavy' in its sibling).

    2026-07-22: delegates to arm_registry.label_for() instead of stripping inline, so this shares ONE
    rule with _display_label / _short_label / the train-graph legends. Kept as a named wrapper because
    the call sites read as 'the short arm tag'; the stripping logic itself now lives in the registry."""
    return label_for(suffix)


def _xb_load_metrics(json_path):
    """eval_metrics.json → {arm_suffix: {metric:(mean, ci_half)}} — strips the backbone prefix so 2B and 1B
    key on the SAME suffix; drops no-data (n_test=—) rows."""
    out = {}
    for r in json.loads(Path(json_path).read_text()):
        if r.get("n_test") in (None, "—", "", 0):
            continue
        enc = r["encoder"]
        for pre in ("vjepa_2_1_vitG_", "vjepa_2_1_vitg_", "vjepa_2_0_vitg_", "vjepa_2_1_"):
            if enc.startswith(pre):
                enc = enc[len(pre):]
                break
        out[enc] = {k: ((r.get(k) or {}).get("mean"), (r.get(k) or {}).get("ci_half"))
                    for k, _, _ in _XB_ALL15}   # load ALL 15 (forest reads its 13 subset; scale reads all)
    return out


def _xb_best(data, keep, key, hi):
    vals = [(s,) + data[s][key] for s in data if keep(s) and data[s][key][0] is not None]
    return (max if hi else min)(vals, key=lambda t: t[1]) if vals else None


def plot_forest(backbones, out_dir, mode="ci", vs="best", stem="forest_plot_ci",
                suptitle=None, sort_rows=True):
    """Per metric: surgery(best-OURs) advantage over a BASELINE. Both baselines count surgery_raw with
    OURs via _xb_is_ours. LAYOUT CONTRACT (visual-audit rules, user 2026-07-12 — see .claude/plotting.md):
      · panels ALWAYS stack VERTICALLY (one per backbone, champion-first) — a portrait figure that fits a
        single AAAI paper column; a side-by-side landscape layout is BANNED (can't fit a 2-column paper).
      · each row names its winner arm in a dedicated RIGHT-hand margin column (» name, arm-registry
        colour) — NEVER the left margin, where it would strike through the metric tick labels.
      · the suptitle is textwrap'd so its width stays INSIDE the figure (no full-width header → no
        wasted flank space); the top margin is sized from the wrapped line count.
      vs='best'   → baseline = the best COMPETITOR arm (toughest bar — includes full-FT).
      vs='frozen' → baseline = the FROZEN backbone only (the paper's stated 'beat frozen' claim).
      mode='ci'   → x = advantage / (95% CI of the difference, sqrt(ci_ours^2+ci_base^2)); green = a
                    SEPARATED win (>=1, right of the dashed 1×CI line) — the statistical view.
      mode='mean' → x = advantage as % of the baseline's mean; green = surgery's MEAN is better (>0)."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    n = len(backbones)
    # suptitle FIRST — its wrapped line count sizes the figure's top margin.
    _base = "frozen" if vs == "frozen" else "the best competitor"
    _sup = suptitle if suptitle else (
        f"Forest plot — does surgery statistically separate from {_base}?   "
        f"(green past dashed line = yes · » names each metric's winner arm)"
        if mode == "ci" else
        f"Forest plot — does surgery's MEAN beat {_base}?   (green = yes · » names each metric's winner arm)")
    _sup = "\n".join(textwrap.fill(seg, width=68) for seg in _sup.split("\n"))
    _nl = _sup.count("\n") + 1
    W, PANEL_H, TITLE_H, BOT_H = 8.6, 6.8, 0.26 * _nl + 0.30, 0.62          # inches
    H = PANEL_H * n + TITLE_H + BOT_H
    fig, axes = plt.subplots(n, 1, figsize=(W, H), squeeze=False)
    for ax, (label, data) in zip(axes.flatten(), backbones):
        if not data:   # None/empty → N/A panel (a backbone with no eval at this scale yet, e.g. 2B FULL)
            ax.text(0.5, 0.5, "N/A\n\nno eval at\nthis scale yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="#9E9E9E", fontweight="bold")
            ax.set_title(label, fontsize=15, fontweight="bold", loc="left")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        rows = []
        for key, dr, lbl in _XB_METRICS:
            hi = dr == "hi"
            bo = _xb_best(data, _xb_is_ours, key, hi)
            if vs == "frozen":
                _fr = data.get("frozen", {}).get(key)
                bc = ("frozen", _fr[0], _fr[1]) if (_fr and _fr[0] is not None) else None
            else:
                bc = _xb_best(data, lambda s: not _xb_is_ours(s), key, hi)
            if not bo or not bc:
                continue
            adv = (bo[1] - bc[1]) if hi else (bc[1] - bo[1])          # +ve = surgery better
            # keep BOTH arm names (best-OURs bo[0] · best-competitor bc[0]) so each row can declare its
            # OWN winner by exact variant name — no blanket "surgery_raw ∈ OURs" title hardcode (2026-07-12).
            if mode == "ci":
                ci = ((bo[2] or 0) ** 2 + (bc[2] or 0) ** 2) ** 0.5 or 1e-9
                rows.append((lbl, adv / ci, bo[0], bc[0]))
            else:
                rows.append((lbl, 100.0 * adv / abs(bc[1]) if bc[1] else 0.0, bo[0], bc[0]))
        if sort_rows:                                                # per-panel best→worst (default); False → fixed _XB_METRICS order
            rows.sort(key=lambda t: t[1])                            # y=0 (bottom) = worst → best at TOP
        fmt = (lambda v: f"{v:.1f}×") if mode == "ci" else (lambda v: f"{v:+.1f}%")
        # xlim/scale FIRST (they only need `rows`) so the draw loop below can flip edge-hugging labels.
        _lt = 1.0 if mode == "ci" else 0.5
        ax.set_xscale("symlog", linthresh=_lt)
        _pos = max([r[1] for r in rows] + [_lt])
        _neg = min([r[1] for r in rows] + [-_lt])
        # positive side gets EXTRA headroom (×6): the value label sits by the marker and must not run
        # into the RHS winner-arm column when the best row hugs the axis edge (visual-audit 2026-07-12).
        ax.set_xlim(min(_neg * 3, -_lt * 3), max(_pos * 6, _lt * 3))
        for y, (lbl, v, ours_s, comp_s) in enumerate(rows):
            won = (v >= 1.0) if mode == "ci" else (v > 0.0)
            col = _XB_OURS_GREEN if won else "#90A4AE"
            if mode == "ci":
                ax.errorbar(v, y, xerr=1, fmt="o", ms=7, color=col, ecolor=col, elinewidth=1.5, capsize=3, zorder=3)
            else:
                ax.plot(v, y, "o", ms=8, color=col, zorder=3)
            # label with a FIXED screen-space gap (offset points) — a data-space offset can't stay legible on
            # the symlog axis (task2 2026-07-01). LIFTED +6pt above the whisker line (VM15: an on-line label
            # gets struck by the whisker; a struck minus reads as PLUS) and FLIPPED to the marker's left for
            # the right-edge cluster (VM16: edge labels crossed the spine into the winner column).
            _edge = v > 0 and v >= 0.5 * _pos
            ax.annotate(fmt(v), (v, y), xytext=((-6, 6) if _edge else (6, 6)), textcoords="offset points",
                        va="bottom", ha=("right" if _edge else "left"), fontsize=11, color=col, fontweight="bold")
            # per-row WINNER declaration (user 2026-07-12): the exact arm that won THIS metric on the mean —
            # v>0 the best-OURs arm, v<0 the best-competitor arm (0 = exact tie). Rendered in a dedicated
            # RIGHT-hand margin column in the winner arm's registry colour (_color_for → same hue as the
            # scorecard bars), so a grey not-separated row still names WHO leads. RHS, never LHS: a left-
            # margin annotation strikes through the metric tick labels (visual-audit mistake #1).
            win_s = ours_s if v > 0 else (comp_s if v < 0 else None)
            wcol = _color_for(f"vjepa_2_1_vitG_{win_s}", 0) if win_s else "#607D8B"
            ax.annotate("» " + (_xb_arm_short(win_s) if win_s else "exact tie"),
                        xy=(1.05, y), xycoords=("axes fraction", "data"), ha="left", va="center",
                        fontsize=9.5, fontweight="bold", fontstyle="italic", color=wcol)
        # (no per-panel "winner arm" column header — it collides with long left-loc panel titles; the
        # suptitle's "» names each metric's winner arm" documents the column instead. visual-audit 2026-07-12)
        # symlog x-axis (task2): ABSOLUTE signed values but NON-uniform spacing — scale+xlim were set ABOVE
        # the draw loop (VM16). Decade-only ticks: bracket the ACTUAL data each side, keep only 0 + 10^k
        # majors, drop the minor subticks (a symmetric margin crammed phantom -1000 ticks). task 2026-07-01.
        _cand = [0] + [10 ** k for k in range(6)] + [-10 ** k for k in range(6)]   # 0 · ±1 · ±10 … ±1e5
        _lo, _hi = ax.get_xlim()
        ax.set_xticks([t for t in _cand if _lo <= t <= _hi])                       # decades within range only
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _p: "0" if v == 0 else ("-" if v < 0 else "") + f"{abs(v):g}"))
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=11, fontweight="bold")   # 11pt bold: full metric names (from json)
        ax.axvline(0, color="#607D8B", lw=1.2, zorder=1)             # null: surgery == baseline
        if mode == "ci":
            ax.axvline(1, color=_XB_OURS_GREEN, ls="--", lw=1.4, zorder=1)
            # right-ANCHORED, BOTTOM-right inside the axes (VM16 saga: left-anchored at x=1 ran past the
            # spine; top-right inside hit the top row's value label; above the axes hit the panel title).
            # Bottom-right is guaranteed empty: rows sort best→worst, so the bottom rows hug the LEFT.
            ax.text(0.985, 0.02, "1×CI · separated →", transform=ax.transAxes, color=_XB_OURS_GREEN,
                    fontsize=10, fontweight="bold", ha="right", va="bottom")
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(label, fontsize=15, fontweight="bold", loc="left")
        _base = "frozen" if vs == "frozen" else "best competitor"
        ax.set_xlabel(f"surgery advantage over {_base}   "
                      + ("(× CI of difference)" if mode == "ci" else f"(% of {_base} mean)"),
                      fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
        ax.tick_params(axis="x", labelsize=11)              # bigger + bold decade ticks (task 2026-07-09)
        for _t in ax.get_xticklabels():
            _t.set_fontweight("bold")
    fig.suptitle(_sup, x=0.5, y=1 - 0.10 / H, va="top", fontsize=12.5, fontweight="bold")
    # margins in INCHES → fractions: left = the metric tick labels · right = the winner-arm column ·
    # top = the wrapped suptitle block · bottom = the last panel's xlabel. Vertical stack ONLY.
    fig.subplots_adjust(top=1 - TITLE_H / H, bottom=BOT_H / H, left=0.30, right=0.775, hspace=0.34)
    save_fig(fig, str(Path(out_dir) / stem))
    plt.close(fig)


def plot_scale_replication(backbones, out_dir):
    """2B-vs-1B rank-preservation scatter per metric (Spearman ρ over the shared arms). Points hug the
    diagonal ORDER when the ranking survives the scale change. Uses the FIRST two backbones (champion, next)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from collections import OrderedDict
    from scipy.stats import spearmanr
    (xlbl, xd), (ylbl, yd) = backbones[0], backbones[1]
    # short '<tag> · <size>' of each compared backbone, derived from the ACTUAL panel labels — the
    # suptitle/callout used to hardcode "2B→1B" regardless of what was compared (VM14-class 2026-07-13).
    _bb_of = lambda lbl: " · ".join(lbl.split(" · ")[:2])
    _duel = f"{_bb_of(xlbl)} → {_bb_of(ylbl)}"
    shared_encs = [s for s in xd if s in yd]
    per = []                              # (rho, dir, label, [encs], xs, ys) — one per metric
    for key, _dir, lbl in _XB_ALL15:
        sh = [s for s in shared_encs if xd[s][key][0] is not None and yd[s][key][0] is not None]
        xs = [xd[s][key][0] for s in sh]
        ys = [yd[s][key][0] for s in sh]
        rho = spearmanr(xs, ys).correlation if len(xs) > 2 else float("nan")
        per.append((rho, _dir, lbl, sh, xs, ys))
    per.sort(key=lambda t: (t[0] if t[0] == t[0] else -9.0), reverse=True)   # ρ DESC, NaN last — task2.2a
    ncol, nrow = 4, 4                     # 16 cells = 15 ρ-sorted panels + 1 legend/callout cell
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.9 * nrow), squeeze=False)
    axf = axes.flatten()
    FAIL = 0.2                            # ρ < 0.2 ⇒ ranking does NOT survive the scale change (task2.2c)
    for ax, (rho, _dir, lbl, sh, xs, ys) in zip(axf, per):
        cols = [_color_for(f"vjepa_2_1_vitG_{s}", 0) for s in sh]
        ax.scatter(xs, ys, c=cols, s=55, edgecolors="white", linewidths=0.6, zorder=3)
        lo, hi = min(xs + ys), max(xs + ys)
        pad = (hi - lo) * 0.08 or 0.01
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color="#90A4AE", lw=1.0, zorder=1)
        tcol = "#C62828" if (rho != rho or rho < FAIL) else ("#2E7D32" if rho >= 0.5 else "#555")
        ax.set_title(f"{lbl} {'↑' if _dir == 'hi' else '↓'}\nρ = {rho:.3f}", fontsize=8, fontweight="bold", color=tcol)   # 8pt: full names fit the small grid panels
        ax.set_xlabel(xlbl, fontsize=8)
        ax.set_ylabel(ylbl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    # 16th cell = colour→encoder legend (task2.2b) + non-replicating callout (task2.2c)
    lg = axf[len(per)]
    lg.axis("off")
    col2encs = OrderedDict()
    for s in shared_encs:
        # arm names via the SHARED _xb_arm_short → identical to the forest winner column (VM14-class).
        col2encs.setdefault(_color_for(f"vjepa_2_1_vitG_{s}", 0), []).append(_xb_arm_short(s))
    handles = [mpatches.Patch(color=c, label=", ".join(encs)) for c, encs in col2encs.items()]
    lg.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=6.5, frameon=False,
              title="encoder colour → arm(s)", title_fontsize=8, labelspacing=0.3, handlelength=1.1)
    fails = [f"{lbl} (ρ={rho:.2f})" for rho, _d, lbl, *_ in per if (rho != rho) or rho < FAIL]
    lg.text(0.0, 0.02, f"[!] does NOT replicate {_duel} (ρ<{FAIL}):\n" +   # ASCII, not ⚠ (tofu in DejaVu — C5)
            ("\n".join("  · " + f for f in fails) if fails else f"  (all {len(per)} replicate)"),
            transform=lg.transAxes, fontsize=8, va="bottom", ha="left", color="#C62828", fontweight="bold")
    for ax in axf[len(per) + 1:]:
        ax.set_visible(False)
    fig.suptitle(f"Scale replication — does the ranking survive {_duel}?   "
                 f"(all {len(per)} metrics, panels sorted by ρ desc · title colour: green ρ≥0.5 replicates · "
                 f"grey 0.2–0.5 inconclusive · red ρ<{FAIL} fails · dashed = identity y=x)",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.94, hspace=0.52, wspace=0.30, left=0.05, right=0.98, bottom=0.05)
    save_fig(fig, str(Path(out_dir) / "scale_replication"))
    plt.close(fig)


def _xb_n_test(json_path):
    """First non-empty n_test in an eval_metrics.json (every encoder shares the test set → one value)."""
    for r in json.loads(Path(json_path).read_text()):
        nt = r.get("n_test")
        if nt not in (None, "—", "", 0):
            try:
                return int(nt)
            except (TypeError, ValueError):
                return None
    return None


_SCALE_TAG = {"poc": "POC 10k", "full": "FULL 116k"}   # corpus display tag — single source for panel titles


def _xb_panel_label(tag, size, mtag, n_test, roster_n=None):
    """The ONE forest panel-title builder — '<ViT tag> · <size> · <SCALE>  (n_test = N)'. Used by BOTH
    cross_backbone_report and scale_forest_report so the same backbone/corpus is titled IDENTICALLY in
    every forest (visual-audit VM14 2026-07-13: the two reports had drifted to 'ViT-g · 1B' vs
    'ViT-g · POC 10k (n_test = 1,825)' for the SAME thing).
    roster_n (VM17): pass the arm count whenever the panel is RESTRICTED to a shared-arm subset — the
    roster changes the best-competitor baseline, so two same-titled panels must not differ silently."""
    head = " · ".join(p for p in (tag, size, _SCALE_TAG[mtag]) if p)
    lbl = f"{head}  (n_test = {n_test:,})" if n_test else f"{head}  (n_test = n/a)"
    return f"{lbl} · shared-arm roster ({roster_n})" if roster_n else lbl


def scale_forest_report(out_dir):
    """POC(10k)-vs-FULL(116k) scale-comparison forest, one figure per backbone. Two panels — POC | FULL —
    share the SAME forest recipe (surgery-best advantage over frozen, the paper claim) AND the SAME metric
    order (sort_rows=False) so a metric lines up across scales. The roster is the UNION of encoders present
    in ANY backbone's FULL eval — so training+evaling a NEW arm at FULL auto-adds it to BOTH panels (auto-
    reflect); POC is filtered to that roster so it is apples-to-apples. A backbone with no FULL eval yet
    (e.g. 2B) renders an N/A FULL panel. n_test (test-clip count) is shown per scale in each panel label."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    known = {pre.rstrip("_") for pre, *_ in _BB_TAG}
    order = {pre.rstrip("_"): i for i, (pre, *_) in enumerate(_BB_TAG)}
    disc = {}   # bb -> {"poc": eval_metrics.json, "full": eval_metrics.json}
    sizes = {}  # bb -> "1B"/"2B" (from the tree name <bb>_<size>) — feeds the shared panel-title builder
    for mtag in ("poc", "full"):
        for mj in Path("outputs").glob(f"{mtag}/*/eval/*/probe_plot/metrics_watch/*/eval_metrics.json"):
            bb, tree = mj.parent.name, mj.relative_to("outputs").parts[1]
            if bb in known and tree.startswith(bb):        # canonical: <bb> is a real backbone in ITS OWN tree
                disc.setdefault(bb, {})[mtag] = mj
                sizes[bb] = tree[len(bb) + 1:]
    if not disc:
        print("  [scale-forest] no per-backbone eval_metrics.json found — nothing to do")
        return None
    full_roster = set()                                    # union of FULL encoders across backbones (auto-reflects)
    for e in disc.values():
        if "full" in e:
            full_roster |= set(_xb_load_metrics(e["full"]))
    full_roster -= _skip_encoders_from_env(sorted(full_roster))   # honour ITER18_SKIP_ARMS
    if not full_roster:
        print("  [scale-forest] no FULL-scale eval_metrics.json yet — nothing to compare, skipping")
        return None
    stems = []
    for bb in sorted(disc, key=lambda b: order.get(b, 99)):
        e = disc[bb]
        tag = next((t for pre, t, _ in _BB_TAG if pre.rstrip("_") == bb), bb)
        poc_mj, full_mj = e.get("poc"), e.get("full")
        poc_data = {s: v for s, v in _xb_load_metrics(poc_mj).items() if s in full_roster} if poc_mj else {}
        full_data = ({s: v for s, v in _xb_load_metrics(full_mj).items() if s in full_roster}
                     if full_mj else None)                 # None → N/A panel (no FULL-scale eval yet, e.g. 2B)
        poc_n = _xb_n_test(poc_mj) if poc_mj else None
        full_n = _xb_n_test(full_mj) if full_mj else None
        _rn = len(full_roster)                             # VM17: both panels are roster-RESTRICTED — say so
        poc_lbl = _xb_panel_label(tag, sizes.get(bb, ""), "poc", poc_n, roster_n=_rn)
        full_lbl = (_xb_panel_label(tag, sizes.get(bb, ""), "full", full_n, roster_n=_rn) if full_data
                    else _xb_panel_label(tag, sizes.get(bb, ""), "full", None)
                    + "\n— no eval at this scale yet")     # 2-line: a 1-line suffix overflowed the canvas (C4)
        panels = [(poc_lbl, poc_data), (full_lbl, full_data)]
        stem = f"scale_poc_vs_full_{bb}"
        # VM18: the "matches forest_plot_best_ci" claim is only true when the FULL panel has data — gate it.
        _match = " · FULL panel matches forest_plot_best_ci" if full_data else ""
        plot_forest(panels, out_dir, mode="ci", vs="best", sort_rows=True, stem=stem,
                    suptitle=(f"Scale {_SCALE_TAG['poc']} → {_SCALE_TAG['full']} · {tag} — does surgery's "
                              f"separation from the best competitor survive the 12x data jump?\n(same "
                              f"encoders both sides · each panel sorted best→worst · green past dashed = "
                              f"separated win · » names each metric's winner arm{_match})"))
        stems.append(stem)
    print(f"  [scale-forest] → {out_dir}/ · {', '.join(s + '.{png,pdf}' for s in stems)} "
          f"(FULL roster {sorted(full_roster)}, auto-reflects new FULL arms)")
    return out_dir


def cross_backbone_report(mtag, out_dir):
    """Discover EVERY canonical per-backbone metrics_watch (eval_metrics.json + eval_scorecard.pdf) under
    outputs/<mtag>/<bb>_<size>/eval/*/probe_plot/metrics_watch/<bb>/ (champion-first), then emit the
    cross-backbone figures into out_dir: forest_plot + scale_replication + the combined scorecard (PDF+PNG)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    known = {pre.rstrip("_") for pre, *_ in _BB_TAG}
    order = {pre.rstrip("_"): i for i, (pre, *_) in enumerate(_BB_TAG)}
    found = []
    for mj in Path("outputs").glob(f"{mtag}/*/eval/*/probe_plot/metrics_watch/*/eval_metrics.json"):
        bb, tree = mj.parent.name, mj.relative_to("outputs").parts[1]
        if bb in known and tree.startswith(bb):        # canonical: <bb> is a real backbone in ITS OWN tree
            found.append((order.get(bb, 99), bb, tree, mj))
    found.sort()
    if not found:
        print("  [cross-plots] no canonical per-backbone eval_metrics.json found — nothing to do")
        return None
    backbones = []
    for _, bb, tree, mj in found:
        tag = next((t for pre, t, _ in _BB_TAG if pre.rstrip("_") == bb), bb)
        size = tree[len(bb) + 1:] if tree.startswith(bb) else ""
        # panel title via the SHARED builder → identical to the scale forest's title for the same
        # backbone/corpus: '<tag> · <size> · <SCALE>  (n_test = N)' (VM14 2026-07-13).
        backbones.append((_xb_panel_label(tag, size, mtag, _xb_n_test(mj)), _xb_load_metrics(mj)))
    # iter19 2026-07-09 (user order): the forest + scale read eval_metrics.json which _mw_dump keeps at
    # FULL (every arm), so hide ITER18_SKIP_ARMS here too (surgery↔surgical aware) — the combined scorecard
    # already hides them via the per-backbone eval_scorecard.pdf regen skip-filtered, so now ALL cross
    # figures show the SAME non-skipped roster instead of forest showing 24 arms next to a 3-arm scorecard.
    _hide = _skip_encoders_from_env([s for _lbl, m in backbones for s in m])
    if _hide:
        backbones = [(lbl, {s: v for s, v in m.items() if s not in _hide}) for lbl, m in backbones]
        print(f"  [cross-plots] ⏷ {sorted(_hide)} hidden from forest/scale (ITER18_SKIP_ARMS)")
    print(f"  [cross-plots] backbones (champion-first): {[b[0] for b in backbones]}")
    init_style()
    plot_forest(backbones, out_dir, mode="ci", vs="best", stem="forest_plot_best_ci")      # vs BEST competitor · stat
    plot_forest(backbones, out_dir, mode="mean", vs="best", stem="forest_plot_best_mean")  # vs BEST competitor · magnitude
    plot_forest(backbones, out_dir, mode="ci", vs="frozen", stem="forest_plot_frozen_ci")  # vs FROZEN — paper claim
    plot_forest(backbones, out_dir, mode="mean", vs="frozen", stem="forest_plot_frozen_mean")
    for _stale in ("forest_plot", "forest_plot_ci", "forest_plot_mean"):   # drop names from the pre-"best" scheme
        for _ext in (".png", ".pdf"):
            (out_dir / f"{_stale}{_ext}").unlink(missing_ok=True)
    if len(backbones) >= 2:
        plot_scale_replication(backbones, out_dir)
    else:
        print("  [cross-plots] scale_replication skipped — need >=2 backbones")
    scale_forest_report(out_dir)   # POC(10k) vs FULL(116k) per-backbone forest (reads BOTH scales, mtag-agnostic)
    combine_scorecards_pdf([mj.parent / "eval_scorecard.pdf" for _, _, _, mj in found],
                           out_dir / "eval_scorecard_combined.pdf")
    print(f"  [cross-plots] → {out_dir}/ · forest_plot_{{best,frozen}}_{{ci,mean}}.{{png,pdf}} · "
          f"scale_replication.{{png,pdf}} · scale_poc_vs_full_<backbone>.{{png,pdf}} · "
          f"eval_scorecard_combined.{{pdf,png}}")
    return out_dir


def regen_metrics_watch(outputs_root: Path, train_root: Path, out_base: Path, mtag: str):
    """Regenerate ALL metrics_watch artifacts into out_base/<BACKBONE>/ from the eval + train
    artifacts under outputs_root — no scripts/ dependency. Data files keep EVERY arm;
    graphs hide ITER18_SKIP_ARMS (same contract as the watch script)."""
    out_dir = out_base / _MW_BACKBONE
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = _mw_skip_arms()
    skip_encs = {_mw_enc_name(_MW_ARM2ENC[a]) for a in skip if a in _MW_ARM2ENC}
    blocks = _mw_train_blocks(train_root, mtag)
    ev = _mw_eval_rows(outputs_root)
    _mw_dump(blocks, ev, out_dir)            # FULL — every arm, incl. skipped
    vis_blocks = [b for b in blocks if b["arm"] not in skip]
    vis_ev = [r for r in ev if r["_enc_full"] not in skip_encs]
    _mw_render_graphs(vis_blocks, vis_ev, out_dir)
    # The 3 registry-derived eval_metrics.json readers (consolidated here so `--metrics-watch-only` is a
    # superset of the retired scripts/iter18_poc_metrics.py figure set): WiSE-FT v2 sweep table + the HONEST
    # paper scorecard + the eyeball-able TCC chart. They read the eval_metrics.json _mw_dump just wrote; under
    # a LIVE partial run a not-yet-evaled arm just shows pending / is skipped. tcc_chart needs the frozen +
    # 3stage-DI + raw rows (FAIL LOUD KeyError otherwise) → guarded so a partial json can't break the watch.
    mj = out_dir / "eval_metrics.json"
    plot_wiseft_sweep_table(mj, out_dir)
    plot_paper_scorecard(mj, out_dir)
    _pv = get_pipeline_config()["plots"]["validity"]   # iter19 §2b: permutation count + seed (single source)
    plot_metric_validity(mj, out_dir, n_perm=_pv["n_perm"], seed=_pv["seed"])  # convergent/discriminant heatmap
    plot_criterion_validity(mj, out_dir / "criterion.csv", _pv["criterion_metric"], out_dir,
                            n_boot=_pv["criterion_n_boot"], seed=_pv["seed"])  # skips until criterion.csv exists
    _tcc_keys = {"vjepa_2_1_frozen", "vjepa_2_1_surgical_3stage_DI_encoder", "vjepa_2_1_surgery_raw_encoder"}
    _have = {x["encoder"] for x in json.loads(mj.read_text())} if mj.exists() else set()
    if _tcc_keys <= _have:
        plot_tcc_chart(mj, out_dir)
    else:
        print(f"  [tcc-chart] skip — partial json missing {sorted(_tcc_keys - _have)} (not evaled yet)")
    if skip:
        print(f"  [metrics-watch] ⏷ {sorted(skip)} hidden from graphs · still in the json/csv")
    print(f"  [metrics-watch] regenerated → {out_dir}/ · "
          f"{{train_trajectories,kept_scorecard,eval_scorecard,wiseft_sweep_table,eval_scorecard_paper,"
          f"tcc_comparison}}.{{png,pdf}} + {{train,eval}}_metrics.{{json,csv}}")


def main():
    p = argparse.ArgumentParser(
        description="m13 — 14-metric bar-with-CI viz for the probe eval suite (hero: §3.3c).")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--action-probe-root", type=Path, default=None)
    p.add_argument("--motion-cos-root",   type=Path, default=None)
    p.add_argument("--future-mse-root",   type=Path, default=None)
    p.add_argument("--taxonomy-root",     type=Path, default=None,
                   help="m12c probe_taxonomy output dir (per_dim_acc.json) — OPTIONAL (graceful).")
    p.add_argument("--predictor-temporal-root", type=Path, default=None,
                   help="m12e output dir (predictor_temporal_per_variant.json) — OPTIONAL.")
    p.add_argument("--encoder-temporal-root",   type=Path, default=None,
                   help="m12f output dir (encoder_temporal_per_variant.json) — OPTIONAL.")
    p.add_argument("--output-dir",        type=Path, default=None,
                   help="m13 eval/ plot dir (required for every mode EXCEPT --combine-scorecards).")
    p.add_argument("--combine-scorecards", nargs="+", default=None, metavar="PDF",
                   help="Standalone: stack these per-backbone eval_scorecard PDFs (in order) into ONE "
                        "combined comparison PDF at --combine-out, then exit. Vector merge (pymupdf); each "
                        "grid keeps its own _BB_LABEL banner. Mirrors the m13_grouped_winner stacked overview.")
    p.add_argument("--combine-out", type=Path, default=None,
                   help="Output path for --combine-scorecards (e.g. "
                        "outputs/poc/probe_plot/metrics_watch/eval_scorecard_combined.pdf).")
    p.add_argument("--cross-plots", action="store_true",
                   help="Standalone: discover both backbones' metrics_watch (eval_metrics.json + "
                        "eval_scorecard.pdf) and emit the cross-backbone forest_plot + scale_replication + "
                        "combined scorecard (PDF/PNG) into outputs/<cross-mode>/probe_plot/metrics_watch/, "
                        "then exit. Colours reuse the scorecard palette (surgery novelty = green).")
    p.add_argument("--cross-mode", default="poc",
                   help="mode tag for --cross-plots discovery (poc|sanity). Default poc.")
    p.add_argument("--reference-hero", action="append", default=None, metavar="BACKBONE=PNG",
                   help="Paste a pre-made per-backbone Δ-vs-frozen hero into eval/<BACKBONE>/ verbatim "
                        "(e.g. a prior-iter champion whose data isn't in THIS run) and include it in the "
                        "combined vertical stack. Repeatable. Path supplied via CLI — never hardcoded.")
    p.add_argument("--metrics-watch-out", type=Path, default=None,
                   help="Regenerate the metrics_watch artifacts (3 figures + train/eval_metrics "
                        "json/csv) into <this dir>/<ITER18_BACKBONE>/ — self-contained, no "
                        "scripts/iter18_* dependency. Requires --outputs-root.")
    p.add_argument("--outputs-root", type=Path, default=None,
                   help="EVAL root — the dir holding probe_action/ probe_motion_cos/ probe_future_mse/ "
                        "predictor_temporal/ encoder_temporal/ (e.g. "
                        "outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k). Required by --metrics-watch-out.")
    p.add_argument("--train-root", type=Path, default=None,
                   help="TRAIN root — the dir holding the per-arm m09 dirs (e.g. "
                        "outputs/poc/vjepa_2_1_vitG_2B/train). Required by --metrics-watch-out. Kept "
                        "SEPARATE from --outputs-root because eval lives under <backbone>/eval/<split>/ "
                        "and train under <backbone>/train/ — one root cannot address both.")
    p.add_argument("--metrics-watch-only", action="store_true",
                   help="Exit after the metrics_watch regeneration (skip the m13 eval/ plots).")
    p.add_argument("--skip-arms", default="",
                   help="Comma/space-separated arm (or encoder) names to HIDE from the GRAPHS — same "
                        "names as the run's --skip-arms (e.g. 'surgery_3stage_DI_head, cassle_encoder'). "
                        "The train/eval csv+json still keep EVERY arm; only the figures drop them. Merged "
                        "into the single-source ITER18_SKIP_ARMS env.")
    add_wandb_args(p)
    args = p.parse_args()
    # iter19 2026-07-09 (user order): fold --skip-arms into the single-source ITER18_SKIP_ARMS env FIRST —
    # BEFORE the --cross-plots / --combine early-returns — so EVERY figure path (cross-backbone forest +
    # scale + combined scorecard AND the per-corpus plots) hides the SAME arms. surgery↔surgical handled
    # downstream by _skip_encoders_from_env / the _mw registry translation.
    if args.skip_arms:                          # CLI --skip-arms ∪ existing ITER18_SKIP_ARMS env (single source)
        _toks = [t for t in re.split(r"[,\s]+", args.skip_arms) if t]
        os.environ["ITER18_SKIP_ARMS"] = " ".join(
            dict.fromkeys(os.environ.get("ITER18_SKIP_ARMS", "").split() + _toks))
    # Standalone: stack per-backbone eval_scorecard PDFs into ONE combined comparison PDF, then exit
    # (no mode flag / --output-dir needed — it reads finished PDFs and renders nothing).
    if args.combine_scorecards:
        if not args.combine_out:
            sys.exit("ERROR: --combine-scorecards requires --combine-out")
        combine_scorecards_pdf(args.combine_scorecards, args.combine_out)
        return
    if args.cross_plots:                       # cross-backbone forest + scale-replication + combined (PDF/PNG)
        cross_backbone_report(args.cross_mode, Path(f"outputs/{args.cross_mode}/probe_plot/metrics_watch"))
        return
    if not (args.SANITY or args.POC or args.FULL):
        sys.exit("ERROR: specify --SANITY, --POC, or --FULL")
    if args.output_dir is None:
        sys.exit("ERROR: --output-dir is required (except for --combine-scorecards)")
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")

    # ── metrics_watch regeneration (self-contained; see the _mw_* section above) ──
    if args.metrics_watch_out is not None:
        if args.outputs_root is None:
            sys.exit("ERROR: --metrics-watch-out requires --outputs-root (the EVAL root, e.g. "
                     "outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k)")
        if args.train_root is None:
            sys.exit("ERROR: --metrics-watch-out requires --train-root (the dir holding the per-arm "
                     "m09 dirs, e.g. outputs/poc/vjepa_2_1_vitG_2B/train). Eval artifacts live under "
                     "<backbone>/eval/<split>/ and train artifacts under <backbone>/train/, so one "
                     "root cannot address both — pass each explicitly (CLAUDE.md no-derived-paths).")
        init_style()
        regen_metrics_watch(args.outputs_root, args.train_root, args.metrics_watch_out, mode.lower())
        # iter19 2026-07-09 (user order): ONE "refresh metrics_watch" now ALSO rebuilds the TOP-LEVEL
        # cross-backbone figures — forest_plot_{best,frozen}_{ci,mean} + scale_replication +
        # eval_scorecard_combined.{pdf,png} — into outputs/<mode>/probe_plot/metrics_watch/, so the live
        # status-pane preview and the §3 finale share ONE recipe and every watch figure tracks the evals
        # finished so far. Partial-tolerant (scale skips <2 backbones; forest uses whatever metrics exist —
        # verified on 1B-only live data). GUARDED: a cross-plot hiccup on a partial state must NOT fail the
        # per-backbone watch that already saved above (same defensive contract as the tcc_chart skip).
        try:
            cross_backbone_report(mode.lower(), Path(f"outputs/{mode.lower()}/probe_plot/metrics_watch"))
        except Exception as _exc:
            print(f"  [cross-plots] skipped on partial data: {type(_exc).__name__}: {_exc}")
        if args.metrics_watch_only:
            return
    elif args.metrics_watch_only:
        sys.exit("ERROR: --metrics-watch-only requires --metrics-watch-out")

    sub_dir = args.output_dir / "eval"
    if sub_dir.exists():
        nfile = sum(1 for _ in sub_dir.rglob("*") if _.is_file())
        print(f"  [m13_eval_plot] wiping eval/ — {nfile} stale file(s)")
        shutil.rmtree(sub_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = sub_dir

    # The 3 legacy headline roots are REQUIRED (FAIL LOUD); temporal/taxonomy are graceful.
    missing = [name for name, val in [
        ("--action-probe-root", args.action_probe_root),
        ("--motion-cos-root",   args.motion_cos_root),
        ("--future-mse-root",   args.future_mse_root),
    ] if val is None]
    if missing:
        sys.exit(f"ERROR: m13 requires: {', '.join(missing)}")

    boot_str = f"{N_BOOTSTRAP // 1000} K bootstrap" if N_BOOTSTRAP >= _K_FMT_MIN else f"{N_BOOTSTRAP} bootstrap"

    wb = init_wandb("m13_eval_plot", mode, config=vars(args), enabled=not args.no_wandb)
    try:
        init_style()
        srcs = {
            "action":   _load_json(args.action_probe_root / artifact("probe_paired_delta"), "Stage 4"),
            "motion":   _load_json(args.motion_cos_root / artifact("probe_motion_cos_paired"), "Stage 7"),
            "future":   _load_json(args.future_mse_root / artifact("probe_future_mse_per_variant"), "Stage 9"),
            "taxonomy": _opt_json(args.taxonomy_root / artifact("per_dim_acc")) if args.taxonomy_root else None,
            "pred":     _opt_json(args.predictor_temporal_root / artifact("predictor_temporal_per_variant"))
                        if args.predictor_temporal_root else None,
            "enc":      _opt_json(args.encoder_temporal_root / artifact("encoder_temporal_per_variant"))
                        if args.encoder_temporal_root else None,
        }
        # Encoder union across all present sources (no hardcoded list).
        encoders = sorted(
            set(srcs["action"].get("by_encoder", {}))
            | set(srcs["motion"].get("by_encoder", {}))
            | {v for v, e in srcs["future"].get("by_variant", {}).items() if isinstance(e, dict)}
        )
        if not encoders:
            sys.exit("FATAL: no encoders found in the action/motion/future JSONs")
        _hide = _skip_encoders_from_env(encoders)   # honor ITER18_SKIP_ARMS in the PLOTS
        if _hide:
            print(f"  [--skip-arms] hiding {len(_hide)} arm(s) from the plots: "
                  f"{sorted(_short_label(e) for e in _hide)}")
            encoders = [e for e in encoders if e not in _hide]
            if not encoders:
                sys.exit("FATAL: all encoders skipped via ITER18_SKIP_ARMS — nothing to plot")

        metrics = _load_all_metrics(srcs, encoders)
        pbar = make_pbar(total=len(_CATALOG), desc="m13_eval_plot", unit="panel")
        n_panels = plot_all_metric_bars(metrics, args.output_dir, boot_str)
        pbar.update(len(_CATALOG))
        pbar.close()
        plotted = sorted(k for k in metrics if _direction_of(k) != "signed")
        signed = sorted(k for k in metrics if _direction_of(k) == "signed")
        print(f"  {n_panels}/{len(_CATALOG)} bar panels written ({len(plotted)} plotted): {plotted}"
              + (f"  ·  signed (CSV-only, not plotted): {signed}" if signed else ""))

        # ── HERO (B1 table-with-CI + B2 Δ-vs-frozen heatmap) — plan §7.3 ──
        # baseline = runtime-discovered 'frozen' encoder (no hardcoded name); needs ≥1 contender.
        frozen = _pick_frozen_ref(encoders)
        # Split: the trained arms + their same-backbone frozen reference = the 'core' hero views.
        # The external image/video baselines (frozen/other) would clutter those views with mostly-N/A
        # predictor cells, so they get a SEPARATE absolute-value scorecard (no Δ) instead.
        core = [e for e in encoders if _arm_family(e) in ("surgery", "pretrain", "ft") or e == frozen]
        frozen_only = [e for e in encoders if _arm_family(e) in ("frozen", "other")]
        # Reference heroes: BACKBONE=PNG to paste verbatim (a prior-iter champion not evaluated here).
        ref_heroes = {}
        for s in (args.reference_hero or []):
            if "=" not in s:
                sys.exit(f"ERROR: --reference-hero must be BACKBONE=PNG, got {s!r}")
            bb, png = s.split("=", 1)
            ref_heroes[bb] = png
        if frozen and len(core) >= _MIN_COMPARABLE:
            # PER-BACKBONE views ONLY — each backbone's arms vs its OWN frozen → eval/<backbone>/.
            # We deliberately do NOT build a cross-backbone combined grid: the backbones have DIFFERENT
            # frozen baselines, so averaging / mixing their Δs is meaningless. The combined overview is a
            # pure VERTICAL IMAGE STACK of these panels (below) — append, never average.
            stacked = []
            for bb in _backbones_present(encoders):
                bb_enc = [e for e in encoders if _backbone_of(e) == bb]
                bb_frozen = _pick_frozen_ref(bb_enc)
                bb_core = [e for e in bb_enc if _arm_family(e) in ("surgery", "pretrain", "ft") or e == bb_frozen]
                if bb_frozen and len(bb_core) >= _MIN_COMPARABLE:
                    print(f"  [per-backbone] {bb} → {args.output_dir.name}/{bb}/  "
                          f"({len(bb_core)} arms vs {_short_label(bb_frozen)})")
                    _emit_hero_suite(metrics, bb_core, bb_frozen, args.output_dir / bb, boot_str)
                    stacked.append(bb)
                else:
                    print(f"  [per-backbone] {bb} — skipped (frozen={bb_frozen}, {len(bb_core)} core arms)")
            # Paste any reference (prior-iter) heroes into their eval/<backbone>/ verbatim, then stack.
            for bb, png in ref_heroes.items():
                src = Path(png)
                if not src.exists():
                    print(f"  [reference-hero] {bb}: {png} not found — skip"); continue
                dst = args.output_dir / bb / "m13_hero_raw_values.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src), str(dst))
                src_pdf = src.with_suffix(".pdf")             # bring the matching .pdf too (both png & pdf)
                if src_pdf.exists():
                    shutil.copy(str(src_pdf), str(dst.with_suffix(".pdf")))
                print(f"  [reference-hero] {bb} ← {png}")
                if bb not in stacked:
                    stacked.append(bb)
            # COMBINED = the per-backbone panels APPENDED VERTICALLY (in _BB_TAG order), NO averaging.
            bb_order = [pre.rstrip("_") for pre, _lt, _st in _BB_TAG]
            _present = [b for b in bb_order if b in stacked]
            _vstack_panels(args.output_dir, _present, "m13_hero_raw_values",
                           "m13_hero_raw_values")
            # COMBINED surgery-vs-pretrain overview = a colourful paired-diff HEATMAP (metrics ×
            # backbones), NOT the sparse stacked forests. Per-backbone forests still live in eval/<bb>/.
            plot_paired_diff_heatmap(metrics, encoders, args.output_dir, boot_str)
        else:
            print(f"  [hero] skipped — needs a 'frozen' baseline + ≥2 core encoders (got {core})")
        # FROZEN-only absolute scorecard (image/video baselines + the V-JEPA frozen reference).
        if len(frozen_only) >= _MIN_COMPARABLE:
            plot_frozen_scorecard(metrics, frozen_only, args.output_dir, boot_str)

        # wandb metric upload — generic prefix + encoder name (NO hardcoded keys).
        wb_metrics = {"n_encoders": len(encoders), "n_metric_panels": n_panels}
        for mkey, md in metrics.items():
            for e, (val, _ci) in md["by_encoder"].items():
                wb_metrics[f"{mkey}__{e}"] = float(val)
        log_metrics(wb, wb_metrics)
        print(f"  Plots written to: {args.output_dir}")
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
