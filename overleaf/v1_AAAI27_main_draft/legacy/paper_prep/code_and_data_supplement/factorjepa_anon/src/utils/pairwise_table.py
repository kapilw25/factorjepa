"""Per-technique head-to-head tables from the eval roll-up `pairwise_deltas` blocks.

The 6 metrics_watch figures (forest / scorecard / scale) only ever show best-OURS vs
best-COMPETITOR. The per-technique numbers — FactorJEPA vs LoRA, vs DoRA, vs Auto-RGN,
vs LP-FT, vs full-FT, vs continual-SSL — are ALREADY computed with paired BCa bootstrap
and sit unused in the stage roll-up JSONs. This module reads them, orients every delta
so "+ = OURS better", labels each arm OURS/COMPETITOR from configs/arm_registry.yaml,
and emits long-form CSV + ASCII + LaTeX tables.

Source files consumed per eval root (all produced by m12a/m12b/m12d/m12e/m12f):
    probe_action/probe_paired_delta.json                  -> act
    probe_motion_cos/probe_motion_cos_paired.json         -> mcos
    probe_future_mse/probe_future_mse_per_variant.json    -> fut
    predictor_temporal/predictor_temporal_per_variant.json-> rollout causal tdist
                                                             teacher_free maskratio order
    encoder_temporal/encoder_temporal_per_variant.json    -> aot tov pace

NOT consumed (shape does not yield one scalar pair-delta): probe_taxonomy/per_dim_acc.json
(15 separate per-dim blocks) and encoder_temporal `tcc` (nested cycle_back/kendalls_tau,
and only 2 pairs were exported).

Direction (higher/lower/signed) comes from configs/metric_names.json — no literal here.
OURS membership mirrors src/m13_eval_plot._xb_is_ours exactly: registry group in
{ours_flagship, ours_head, improvement} PLUS the explicit surgery_raw_encoder exception.

USAGE (every path arg required — CLAUDE.md no-default rule):
    python -u src/utils/pairwise_table.py \
        --eval-root  POC-2B=outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k \
        --eval-root  POC-1B=outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k \
        --eval-root  FULL-1B=outputs/full/vjepa_2_1_vitg_1B/eval/full \
        --metric-names configs/metric_names.json \
        --arm-registry configs/arm_registry.yaml \
        --headline act,fut,causal,mcos,maskratio \
        --ours-arm surgical_3stage_DI_encoder \
        --ours-arm surgery_raw_encoder \
        --out-dir overleaf/2026___FactorJEPA_AAAI/paper_prep/results_tables \
        2>&1 | tee logs/pairwise_table_$(date +%Y%m%d_%H%M%S).log
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

# ── stage file -> metric keys it carries ─────────────────────────────────────
# (relative path, mode) where mode says how to reach the {metric: block} mapping.
_FLAT = "flat"      # file root IS one metric block ({by_*, pairwise_deltas})
_NEST = "nested"    # file root is {"metrics": {name: block}}
_SOURCES = [
    ("probe_action/probe_paired_delta.json", _FLAT, {"act": None}),
    ("probe_motion_cos/probe_motion_cos_paired.json", _FLAT, {"mcos": None}),
    ("probe_future_mse/probe_future_mse_per_variant.json", _FLAT, {"fut": None}),
    ("predictor_temporal/predictor_temporal_per_variant.json", _NEST,
     {"rollout": "rollout", "causal": "causal", "tdist": "tdist",
      "teacher_free": "teacher_free", "maskratio": "maskratio", "order": "order"}),
    ("encoder_temporal/encoder_temporal_per_variant.json", _NEST,
     {"aot": "aot", "tov": "tov", "pace": "pace"}),
]

# Each roll-up writer chose its own field names. Map them to one schema; anything
# outside these alternatives is a schema change and must CRASH, not be defaulted away.
_DELTA_KEYS = ("delta_pp", "delta_mean")
_LO_KEYS = ("ci_lo_pp", "delta_ci_lo")
_HI_KEYS = ("ci_hi_pp", "delta_ci_hi")
_N_KEYS = ("n_shared", "n")
_SCALE_PP = {"delta_pp"}          # these files report percentage POINTS, not raw units


def _pick(d: dict, alts: tuple, where: str):
    for k in alts:
        if k in d:
            return k, d[k]
    raise RuntimeError(
        f"{where}: none of {alts} present in pairwise entry (keys={sorted(d)}). "
        "The roll-up schema changed — fix the mapping instead of silently skipping."
    )


def load_metric_dirs(metric_names_json: Path) -> dict:
    """{key: 'higher'|'lower'|'signed'} straight from the single-source registry."""
    reg = json.loads(Path(metric_names_json).read_text())["metrics"]
    return {k: v["dir"] for k, v in reg.items()}


def load_metric_labels(metric_names_json: Path) -> dict:
    reg = json.loads(Path(metric_names_json).read_text())["metrics"]
    return {k: v["name"] for k, v in reg.items()}


def load_sides(arm_registry_yaml: Path) -> dict:
    """{encoder_token: 'OURS'|'COMP'} — mirrors m13_eval_plot._xb_is_ours."""
    arms = yaml.safe_load(Path(arm_registry_yaml).read_text())["arms"]
    ours_groups = {"ours_flagship", "ours_head", "improvement"}
    sides = {}
    for spec in arms.values():
        enc = spec["encoder"]
        sides[enc] = "OURS" if spec["group"] in ours_groups else "COMP"
    sides["surgery_raw_encoder"] = "OURS"   # explicit exception, same as the plot code
    return sides


def strip_backbone(enc: str) -> str:
    for pre in ("vjepa_2_1_vitG_", "vjepa_2_1_vitg_", "vjepa_2_0_vitg_", "vjepa_2_1_"):
        if enc.startswith(pre):
            return enc[len(pre):]
    return enc


def _iter_blocks(root: Path):
    """Yield (metric_key, block) for every metric this eval root exports."""
    for rel, mode, keymap in _SOURCES:
        p = root / rel
        if not p.exists():
            print(f"    [skip] {rel} absent")
            continue
        doc = json.loads(p.read_text())
        if mode == _FLAT:
            (mk,) = keymap
            yield mk, doc, rel
        else:
            for mk, sub in keymap.items():
                if sub in doc["metrics"]:
                    yield mk, doc["metrics"][sub], f"{rel}::{sub}"


def extract(root: Path, scale: str, dirs: dict, sides: dict) -> list:
    """Long-form rows: one per (metric, ordered arm pair), oriented + side-labelled."""
    rows = []
    for mk, block, prov in _iter_blocks(root):
        pw = block.get("pairwise_deltas")
        if not pw:
            print(f"    [skip] {prov}: no pairwise_deltas")
            continue
        direction = dirs[mk]                       # KeyError = metric not in registry -> loud
        for key, e in pw.items():
            if "_minus_" not in key:
                raise RuntimeError(f"{prov}: pair key {key!r} lacks '_minus_'")
            a_raw, b_raw = key.split("_minus_", 1)
            a, b = strip_backbone(a_raw), strip_backbone(b_raw)
            dk, delta = _pick(e, _DELTA_KEYS, f"{prov}:{key}")
            _, lo = _pick(e, _LO_KEYS, f"{prov}:{key}")
            _, hi = _pick(e, _HI_KEYS, f"{prov}:{key}")
            _, n = _pick(e, _N_KEYS, f"{prov}:{key}")
            unit = "pp" if dk in _SCALE_PP else "raw"
            # orient so + = A better than B; 'signed' metrics have no better direction
            sgn = 1.0 if direction == "higher" else (-1.0 if direction == "lower" else 0.0)
            if sgn:
                adv, adv_lo, adv_hi = sgn * delta, sgn * min(lo, hi), sgn * max(lo, hi)
            else:
                adv = adv_lo = adv_hi = None
            rows.append({
                "scale": scale, "metric": mk, "direction": direction, "unit": unit,
                "arm_a": a, "arm_b": b,
                "side_a": sides.get(a, "REF" if a == "frozen" else "?"),
                "side_b": sides.get(b, "REF" if b == "frozen" else "?"),
                "n": n,
                "raw_delta_a_minus_b": delta, "raw_ci_lo": lo, "raw_ci_hi": hi,
                "a_advantage": adv, "a_adv_ci_lo": adv_lo, "a_adv_ci_hi": adv_hi,
                "separated": (None if adv is None else bool(adv_lo > 0 or adv_hi < 0)),
                "p_value": e.get("p_value"), "source": prov,
            })
    return rows


def lookup(rows, scale, metric, ours, comp):
    """The (ours vs comp) row for one metric, flipped if the file stored comp_minus_ours."""
    for r in rows:
        if r["scale"] != scale or r["metric"] != metric:
            continue
        if r["arm_a"] == ours and r["arm_b"] == comp:
            return r
        if r["arm_a"] == comp and r["arm_b"] == ours:
            f = dict(r)
            for k in ("raw_delta_a_minus_b", "raw_ci_lo", "raw_ci_hi"):
                f[k] = -r[k]
            if r["a_advantage"] is not None:
                f["a_advantage"] = -r["a_advantage"]
                f["a_adv_ci_lo"], f["a_adv_ci_hi"] = -r["a_adv_ci_hi"], -r["a_adv_ci_lo"]
            f["arm_a"], f["arm_b"] = ours, comp
            f["side_a"], f["side_b"] = r["side_b"], r["side_a"]
            return f
    return None


def _fmt(r):
    if r is None:
        return "n/a"
    if r["a_advantage"] is None:
        return "signed"
    u = " pp" if r["unit"] == "pp" else ""
    return f"{r['a_advantage']:+.4f}{u}{'*' if r['separated'] else ''}"


def ascii_table(rows, scale, ours, comps, headline, labels, n_of):
    out = [f"FactorJEPA arm: {ours}      scale: {scale}      n_test = {n_of}",
           "+ = OURS better · * = 95% BCa CI of the paired difference excludes 0"]
    w = 34
    hdr = f"{'competitor technique':<{w}}" + "".join(f"{labels[m][:17]:>19}" for m in headline)
    out += ["-" * len(hdr), hdr, "-" * len(hdr)]
    for c in comps:
        line = f"{c:<{w}}"
        for m in headline:
            line += f"{_fmt(lookup(rows, scale, m, ours, c)):>19}"
        out.append(line)
    out.append("-" * len(hdr))
    return "\n".join(out)


def latex_table(rows, scale, ours, comps, headline, labels, n_of, caption, label):
    esc = lambda s: s.replace("_", r"\_")
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.15}",
         f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * len(headline) + "}", r"\toprule",
         r"\textbf{Competitor technique} & "
         + " & ".join(r"\textbf{" + esc(labels[m]) + "}" for m in headline) + r" \\",
         r"\midrule"]
    for c in comps:
        cells = []
        for m in headline:
            r = lookup(rows, scale, m, ours, c)
            if r is None or r["a_advantage"] is None:
                cells.append("--")
            else:
                u = r"\,pp" if r["unit"] == "pp" else ""
                v = f"{r['a_advantage']:+.4f}{u}"
                # bold ONLY a separated WIN. A separated LOSS gets \underline — bolding it
                # too would read as "this is the good cell" (the sign is easy to miss at
                # \scriptsize). Ties/overlapping CIs stay plain.
                if r["separated"]:
                    cells.append((r"\textbf{" if r["a_advantage"] > 0 else r"\underline{") + v + "}")
                else:
                    cells.append(v)
        L.append(f"{esc(c)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}%", "}",
          r"\vspace{-0.4em}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="per-technique head-to-head tables from pairwise_deltas")
    ap.add_argument("--eval-root", action="append", required=True,
                    help="SCALE=path/to/eval/root (repeatable)")
    ap.add_argument("--metric-names", required=True, help="configs/metric_names.json")
    ap.add_argument("--arm-registry", required=True, help="configs/arm_registry.yaml")
    ap.add_argument("--headline", required=True, help="comma-separated metric keys for the tables")
    ap.add_argument("--ours-arm", action="append", required=True,
                    help="OURS arm to tabulate (repeatable; one table block each)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    dirs, labels = load_metric_dirs(a.metric_names), load_metric_labels(a.metric_names)
    sides = load_sides(a.arm_registry)
    headline = [m.strip() for m in a.headline.split(",") if m.strip()]
    for m in headline:
        if m not in dirs:
            raise RuntimeError(f"--headline {m!r} is not in {a.metric_names}")

    rows, n_of = [], {}
    for spec in a.eval_root:
        if "=" not in spec:
            raise RuntimeError(f"--eval-root must be SCALE=path, got {spec!r}")
        scale, path = spec.split("=", 1)
        print(f"  [{scale}] {path}")
        got = extract(Path(path), scale, dirs, sides)
        rows += got
        n_of[scale] = max((r["n"] for r in got), default=0)
        print(f"    {len(got)} pair-rows")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with (out / "pairwise_all.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    oc = [r for r in rows if {r["side_a"], r["side_b"]} == {"OURS", "COMP"}]
    with (out / "ours_vs_technique.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(oc)
    vf = [r for r in rows if "frozen" in (r["arm_a"], r["arm_b"])]
    with (out / "arm_vs_frozen.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(vf)

    scales = [s.split("=", 1)[0] for s in a.eval_root]
    txt, tex = [], []
    for ours in a.ours_arm:
        for scale in scales:
            comps = sorted({r["arm_b"] for r in rows
                            if r["scale"] == scale and r["arm_a"] == ours and r["side_b"] == "COMP"}
                           | {r["arm_a"] for r in rows
                              if r["scale"] == scale and r["arm_b"] == ours and r["side_a"] == "COMP"})
            if not comps:
                txt.append(f"\n{ours} @ {scale}: no competitor pairs\n")
                continue
            txt.append(ascii_table(rows, scale, ours, comps, headline, labels, n_of[scale]) + "\n")
            tex.append(latex_table(
                rows, scale, ours, comps, headline, labels, n_of[scale],
                caption=(rf"\textbf{{Per-technique head-to-head at {scale}.}} Paired BCa "
                         rf"difference (\texttt{{{ours.replace('_', chr(92) + '_')}}} minus each "
                         r"competitor), oriented so positive favours ours. \textbf{Bold} = a 95\% "
                         r"CI excluding zero in our favour; \underline{underline} = a 95\% CI "
                         r"excluding zero against us; plain = CIs overlap."),
                label=f"tab:h2h_{ours}_{scale}".replace("-", "_")) + "\n")
    (out / "per_technique_tables.txt").write_text("\n".join(txt))
    (out / "per_technique_tables.tex").write_text("\n".join(tex))

    print(f"\n  wrote {len(rows)} pair-rows ({len(oc)} OURSxCOMP, {len(vf)} vs-frozen) -> {out}/")
    for f in sorted(out.iterdir()):
        print(f"    {f.name:<32} {f.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    sys.exit(main())
