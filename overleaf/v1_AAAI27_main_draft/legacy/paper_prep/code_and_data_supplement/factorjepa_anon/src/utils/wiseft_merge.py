"""
WiSE-FT weight-space merge — lever #1 of plan_outperform_FT.md (iter18 2026-06-13).

Produces a NEW encoder by linearly interpolating a fine-tuned (surgery) encoder with the
FROZEN base encoder, WITHOUT any retraining:

        θ*(α) = (1 − α) · θ_frozen  +  α · θ_surgery          α ∈ [0, 1]

This recovers Frozen-level temporal structure (TCC) that fine-tuning erodes, while keeping
surgery's prediction gains — the interpolation curve is typically convex, so a mid α can
Pareto-dominate either endpoint. Pick α on the VAL split, then that ckpt IS the new arm.

Gold standard: Wortsman et al., "Robust fine-tuning of zero-shot models" (WiSE-FT), CVPR 2022
(arXiv:2109.01903); generalised by "Model soups" (ICML 2022, arXiv:2203.05482).

ADDITIVE: writes ONLY new files under a NEW arm dir — the surgery + frozen checkpoints are
read-only, so previously-trained endpoints are untouched and remain comparable in eval.

USAGE:
  # inspect both checkpoints' state-dict structure first (no write):
  python -u src/utils/wiseft_merge.py --inspect \
      --frozen-ckpt checkpoints/vjepa2_1_vitG_384.pt \
      --surgery-ckpt outputs/poc/vjepa_2_1_vitG/m09c_surgery_3stage_DI_encoder/student_encoder.pt

  # sweep α and write one candidate per α (for val-selection):
  python -u src/utils/wiseft_merge.py \
      --frozen-ckpt  checkpoints/vjepa2_1_vitG_384.pt \
      --surgery-ckpt outputs/poc/vjepa_2_1_vitG/m09c_surgery_3stage_DI_encoder/student_encoder.pt \
      --out-dir      outputs/poc/vjepa_2_1_vitG/m09c_surgical_3stage_DI_wiseft_encoder \
      --alphas 0.5 0.6 0.7 0.8

  # write the single chosen α as the arm's exported encoder:
  python -u src/utils/wiseft_merge.py --alpha 0.7 ... --out-dir <arm dir>   # → <arm dir>/student_encoder.pt
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

# Common containers a checkpoint's parameter dict may hide under.
_STATE_KEYS = ("state_dict", "model", "encoder", "module", "model_state_dict", "student_state_dict")


def _find_state_dict(obj, label: str):
    """Return (mapping, container_key) — container_key=None for a raw state_dict. FAIL LOUD if not found."""
    if isinstance(obj, dict):
        # raw state_dict: dict whose values are mostly tensors
        tensor_vals = sum(1 for v in obj.values() if torch.is_tensor(v))
        if tensor_vals >= max(1, len(obj) // 2):
            return obj, None
        for k in _STATE_KEYS:
            if k in obj and isinstance(obj[k], dict):
                return obj[k], k
    sys.exit(f"FATAL: could not locate a parameter state_dict inside {label} "
             f"(top-level type={type(obj).__name__}, keys={list(obj)[:8] if isinstance(obj, dict) else 'n/a'}). "
             f"Add the container key to _STATE_KEYS.")


def _strip(prefixes, sd: dict) -> dict:
    """Drop a leading module prefix (e.g. 'backbone.') PER-KEY so a frozen base with MIXED
    prefixes (backbone.* encoder + predictor.*) aligns with bare surgery encoder keys."""
    out = {}
    for k, v in sd.items():
        changed = True
        while changed:                      # frozen base = DOUBLE-prefixed 'module.backbone.X'
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
                    break
        out[k] = v
    return out


def _load(path: Path, label: str):
    if not path.exists():
        sys.exit(f"FATAL: {label} not found: {path}")
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    sd, key = _find_state_dict(obj, label)
    return _strip(("module.", "encoder.", "backbone."), sd), obj, key


def _common_float_keys(a: dict, b: dict):
    """Keys present + same-shape + floating in BOTH — only these get interpolated."""
    inter, skipped = [], []
    for k in a:
        if k not in b:
            skipped.append((k, "missing in surgery"))
        elif a[k].shape != b[k].shape:
            skipped.append((k, f"shape {tuple(a[k].shape)} vs {tuple(b[k].shape)}"))
        elif not torch.is_floating_point(a[k]):
            skipped.append((k, "non-float (copied from surgery)"))
        else:
            inter.append(k)
    return inter, skipped


def _blend(fro: dict, sur: dict, inter, al: float) -> dict:
    """α·surgery + (1−α)·frozen on the shared float keys; non-shared keys/buffers kept from surgery."""
    m = dict(sur)
    for k in inter:
        m[k] = (1.0 - al) * fro[k].float() + al * sur[k].float()
    return m


def main():
    ap = argparse.ArgumentParser(description="WiSE-FT weight-space merge (frozen × surgery).")
    ap.add_argument("--frozen-ckpt",  type=Path, required=True, help="base FROZEN encoder ckpt")
    ap.add_argument("--surgery-ckpt", type=Path, required=True, help="fine-tuned surgery student_encoder.pt")
    ap.add_argument("--surgery-pred-ckpt", type=Path, default=None,
                    help="surgery arm's m09*_ckpt_best.pt — ALSO blend its predictor and write out/m09c_ckpt_best.pt "
                         "so the WiSE-FT arm reports prediction metrics (future-MSE/causal), not just encoder ones")
    ap.add_argument("--out-dir",      type=Path, default=None, help="arm output dir (required unless --inspect)")
    ap.add_argument("--alpha",  type=float, default=None, help="single α → writes <out-dir>/student_encoder.pt")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="α sweep → writes <out-dir>/alpha<α>/student_encoder.pt per α (for val-selection)")
    ap.add_argument("--inspect", action="store_true", help="report state-dict alignment, write nothing")
    a = ap.parse_args()

    fro, _fro_obj, _fro_key = _load(a.frozen_ckpt, "frozen-ckpt")
    sur, sur_obj, sur_key = _load(a.surgery_ckpt, "surgery-ckpt")
    inter, skipped = _common_float_keys(fro, sur)
    print(f"  [wiseft] frozen={len(fro)} keys · surgery={len(sur)} keys · "
          f"interpolated={len(inter)} · skipped={len(skipped)}")
    for k, why in skipped[:8]:
        print(f"    skip {k}: {why}")
    if not inter:
        sys.exit("FATAL: 0 interpolable keys — the two checkpoints don't align (see skips above).")

    # optional predictor blend — Stage 8/8b reads the predictor from <arm>/m09c_ckpt_best.pt, so without this
    # the WiSE-FT arm carries no predictor and future-MSE/causal stay blank (the original encoder-only flaw).
    pred_fro = pred_sur = pred_inter = None
    if a.surgery_pred_ckpt is not None:
        if not (isinstance(_fro_obj, dict) and isinstance(_fro_obj.get("predictor"), dict)):
            sys.exit("FATAL: --frozen-ckpt carries no 'predictor' dict — cannot merge the predictor.")
        pred_fro = _strip(("module.", "encoder.", "backbone."), _fro_obj["predictor"])
        pso = torch.load(str(a.surgery_pred_ckpt), map_location="cpu", weights_only=False, mmap=True)
        if not (isinstance(pso, dict) and isinstance(pso.get("predictor"), dict)):
            top = list(pso)[:6] if isinstance(pso, dict) else type(pso).__name__
            sys.exit(f"FATAL: --surgery-pred-ckpt has no 'predictor' dict (top={top}).")
        pred_sur = _strip(("module.", "encoder.", "backbone."), pso["predictor"])
        pred_inter, pred_skip = _common_float_keys(pred_fro, pred_sur)
        print(f"  [wiseft] predictor: frozen={len(pred_fro)} · surgery={len(pred_sur)} · "
              f"interpolated={len(pred_inter)} · skipped={len(pred_skip)}")
        if not pred_inter:
            sys.exit("FATAL: 0 interpolable predictor keys — frozen/surgery predictors don't align.")

    if a.inspect:
        print("  [wiseft] --inspect only; nothing written.")
        return

    if a.out_dir is None:
        sys.exit("FATAL: --out-dir required unless --inspect")
    alphas = a.alphas if a.alphas is not None else [a.alpha if a.alpha is not None else 0.7]

    for al in alphas:
        if not 0.0 <= al <= 1.0:
            sys.exit(f"FATAL: alpha {al} out of [0,1]")
        merged = _blend(fro, sur, inter, al)                # encoder blend (starts from surgery → keeps buffers)
        # re-wrap in surgery's container (e.g. student_state_dict) so the eval loader reads the
        # merged ckpt byte-identically to every other arm's student_encoder.pt.
        payload = merged if sur_key is None else {**sur_obj, sur_key: merged}
        sub = a.out_dir if (a.alphas is None) else (a.out_dir / f"alpha{al:g}")
        sub.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(sub / "student_encoder.pt"))
        line = f"  [wiseft] α={al:g} → student_encoder.pt ({len(inter)} enc keys)"
        # WiSE-FT convention: the blended model reuses the FINE-TUNED head. Surgery trained a motion_aux
        # head (sits next to --surgery-ckpt); carry it over so eval gets head-augmented future_mse instead
        # of the encoder-only fallback. Transferred head (fit on surgery feats) — footnote in the paper.
        _aux = a.surgery_ckpt.parent / "motion_aux_head.pt"
        if _aux.exists():
            shutil.copy2(_aux, sub / "motion_aux_head.pt")
            line += " + motion_aux_head.pt"
        if pred_inter is not None:
            # combined ckpt Stage 8/8b reads: encoder under 'student', predictor under 'predictor'
            merged_pred = _blend(pred_fro, pred_sur, pred_inter, al)
            torch.save({"student": merged, "predictor": merged_pred, "step": 0,
                        "best_metric": float(al), "is_full": True, "has_optimizer": False},
                       str(sub / "m09c_ckpt_best.pt"))
            line += f" + m09c_ckpt_best.pt ({len(pred_inter)} pred keys)"
        print(line + f"  → {sub}")


if __name__ == "__main__":
    main()
