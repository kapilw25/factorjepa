#!/usr/bin/env python3
"""m14 v2 — VISUAL DEMO of the headline eval metrics as a real VIDEO EDIT (iter20 E1+E2).

v2 format (user 2026-07-12, `driving1.png` style): synchronized PANELS of the SAME street clip
playing at native fps — `Original | model input (real black tiles) | report card per model
(green=right, red=wrong painted on the hidden patches) | score ticker` — one scene per metric:

  🅰️ dense features   Original ∥ PCA→RGB feature video ("same colour = tracked as the same thing")
  🅱️ future-frame MSE random 3-D blocks hidden (the m12d eval mask); report card on those patches
  🅲️ causal L1        first temporal half plays normally, second half fully black → report card
  🅳️ mask-ratio       the SAME clip plays 4 passes at 30→90% hidden; the error line's SLOPE = metric
  🅴️ motion-cos       4 clips play together; link-line thickness = how alike the model finds them

`--scenes C` renders ONE metric only (user option-2 workflow 2026-07-13: perfect the strongest
scene — causal — on [FROZEN vs OURS] first; the other scenes unlock after visual sign-off).

HONESTY CONTRACT: the model predicts DESCRIPTIONS (latent vectors), never pixels — the demo paints
its per-patch report card onto the REAL frames. The mask IS the edit: the black tiles are genuinely
what the model receives (on its 16 eval frames). Display panels run native fps; the MODEL forwards
stay eval-identical (pipeline.yaml probe.num_frames, exact eval preprocessing).

Metric cores are IMPORTED from utils.predictor_eval / utils.pt_causal / utils.pt_maskratio and the
demo's numbers are asserted equal to pt_*.compute (FATAL parity guards — no re-implementation drift).

Gold-standard refs:
  https://github.com/facebookresearch/vjepa2     (PCA dense-feature videos)
  https://github.com/MCG-NJU/VideoMAE            (original ∥ masked ∥ result triptych demos)
  https://arxiv.org/abs/2404.08471               (V-JEPA — latents→pixels needs an EXTRA decoder)

Decode substitution (documented): PyAV is absent in the demo venv (venv_denseworld) — frames come
from the system ffmpeg CLI (all frames → uniform indices for the model, native-fps for display),
then the EXACT eval preprocessing utils.frozen_features.resize_and_normalize for model input.

USAGE (RTX 3060 12 GB · bf16 · models loaded sequentially):
    source venv_denseworld/bin/activate
    set -o pipefail && PYTHONPATH=src python -u src/m14_metric_demo.py \\
        --ckpt "FROZEN 2.1=checkpoints/vjepa2_1_vitg_384.pt" \\
        --ckpt "OURS diheavy=outputs/full/vjepa_2_1_vitg_1B/train/m09c_surgery_3stage_DI_diheavy_encoder/m09c_ckpt_best.pt" \\
        --model-config configs/model/vjepa2_1_vitg.yaml \\
        --demo-config configs/demo.yaml \\
        --clips-dir data/demo_clips \\
        --output-dir outputs/demo/metric_visual \\
        --scenes C \\
        2>&1 | tee logs/m14_demo_v2_$(date +%Y%m%d_%H%M%S).log
    (--scenes all = every scene · --sanity-one-clip = E4 smoke on 1 clip)
"""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from tqdm import tqdm

from utils.config import get_pipeline_config
from utils.frozen_features import resize_and_normalize
from utils.demo_video import (assemble_frames, decode_all_frames, load_token_decoder,
                              resize_center_crop_uint8)
from utils.predictor_eval import (
    PT_MASK_RATIOS, PT_SEED, build_mask_gen, expand_mask, load_encoder_predictor,
    masked_predict_l1, perclip_slope, temporal_token_idx, to_pixel, token_grid)
from utils import pt_causal, pt_maskratio

# VM7: ONE source for every displayed metric name — scene titles, ticker labels, verdict rows and
# demo_metrics.json keys all read THIS dict.
_METRIC_NAME = {
    "A": "dense features (PCA->RGB)",
    "B": "future-frame MSE (latent L1)",
    "C": "causal future-block L1",
    "D": "mask-ratio robustness slope",
    "E": "motion-cosine separation",
    "W": "who imagines the future better?",   # triptych: decoded PREDICTED latents (m15 decoders)
}
_HONESTY = ("pixels from an EXTRA decoder trained by us - V-JEPA predicts descriptions, not pictures")
_BG = (16, 20, 24)
_FG = (236, 239, 241)
_SUB = (176, 190, 197)
_ACC = (128, 203, 196)
_MODEL_COLORS = ((144, 164, 174), (102, 187, 106))     # model 1 grey · model 2 (OURS) green


# ── Token-grid helpers (decode helpers live in utils.demo_video — shared with m15) ──

def tokens_to_grid(values: np.ndarray, idx: np.ndarray, num_frames: int) -> np.ndarray:
    Tp, Hp, Wp, S = token_grid(num_frames)
    flat = np.full(Tp * S, np.nan, dtype=np.float32)
    flat[idx] = values
    return flat.reshape(Tp, Hp, Wp)


def pca_rgb(tokens: np.ndarray, num_frames: int, n_comp: int = 3) -> np.ndarray:
    """(N,D) → (Tp,Hp,Wp,3) in [0,1] — the vjepa2-repo dense-feature viz."""
    Tp, Hp, Wp, _ = token_grid(num_frames)
    X = tokens - tokens.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:n_comp].T
    P = (P - P.min(0)) / (P.max(0) - P.min(0) + 1e-8)
    return P.reshape(Tp, Hp, Wp, n_comp)


# ── Per-model metric/visual payload (models loaded SEQUENTIALLY — 12 GB-safe) ────

def compute_payload(label, ckpt, model_config, batch, keys, heroes, num_frames, shared_masks,
                    scenes):
    """Load ONE model, compute the needed visual + metric ingredients, free the GPU."""
    print(f"[m14] loading {label} ({ckpt}) …")
    encoder, predictor, _ = load_encoder_predictor(str(ckpt), num_frames, model_cfg=str(model_config))
    embed_dim = yaml.safe_load(Path(model_config).read_text())["model"]["embed_dim"]
    Tp, Hp, Wp, S = token_grid(num_frames)
    n_tok = Tp * S
    pl = {"label": label, "feats_final": {}, "pooled": {}, "hero": {}}

    if {"A", "E"} & scenes:
        for i, k in enumerate(tqdm(keys, desc=f"encode[{label}]")):
            pixel = to_pixel(batch[i:i + 1])
            with torch.no_grad():
                h = encoder(pixel)
                if isinstance(h, (list, tuple)):
                    h = torch.cat(list(h), dim=-1)
            pl["feats_final"][k] = h[0].float().cpu().numpy()[:, -embed_dim:]
            pl["pooled"][k] = pl["feats_final"][k].mean(0)
            del h, pixel
            torch.cuda.empty_cache()

    for k in tqdm(heroes, desc=f"metrics[{label}]"):
        i = keys.index(k)
        pixel = to_pixel(batch[i:i + 1])
        hero = {}

        if "B" in scenes:   # Scene B — future-frame MSE (m12d random-block mask, SAME across models)
            m_enc, m_pred = shared_masks[k]
            l1_b, out_t, tgt_t = masked_predict_l1(encoder, predictor, pixel, m_enc, m_pred)
            err = (out_t - tgt_t).abs().mean(-1)[0].float().cpu().numpy()
            pred_idx = m_pred[0].cpu().numpy()
            hero["B_err"] = tokens_to_grid(err, pred_idx, num_frames)
            hid = np.zeros(n_tok, bool); hid[pred_idx] = True
            hero["B_hidden"] = hid.reshape(Tp, Hp, Wp)
            hero["B_l1"] = float(l1_b[0])

        if {"C", "W"} & scenes:   # Scene C/W — causal future-block L1 (EXACT pt_causal mask)
            half = Tp // 2
            m_enc_c = expand_mask(temporal_token_idx(num_frames, range(0, half)), 1)
            m_pred_c = expand_mask(temporal_token_idx(num_frames, range(half, Tp)), 1)
            l1_c, out_c, tgt_c = masked_predict_l1(encoder, predictor, pixel, m_enc_c, m_pred_c)
            if "W" in scenes:
                # the PREDICTOR's latents for the hidden half — last-embed_dim slice of the concat
                # prediction = its estimate of the FINAL encoder layer (what the m15 decoder inverts).
                hero["W_pred"] = out_c[0][:, -embed_dim:].float().cpu().numpy()
            err_c = (out_c - tgt_c).abs().mean(-1)[0].float().cpu().numpy()
            idx_c = m_pred_c[0].cpu().numpy()
            hero["C_err"] = tokens_to_grid(err_c, idx_c, num_frames)
            hid_c = np.zeros(n_tok, bool); hid_c[idx_c] = True
            hero["C_hidden"] = hid_c.reshape(Tp, Hp, Wp)
            hero["C_l1"] = float(l1_c[0])
            hero["C_half"] = half
            hero["C_curve"] = np.nanmean(hero["C_err"][half:].reshape(Tp - half, -1), axis=1)
            causal_ref = pt_causal.compute(encoder, predictor, batch[i:i + 1], num_frames)
            if not np.allclose(causal_ref[0], l1_c[0], rtol=0.05):
                sys.exit(f"FATAL[{label}]: demo causal L1 {l1_c[0]:.5f} != pt_causal {causal_ref[0]:.5f}")

        if "D" in scenes:   # Scene D — mask-ratio robustness (EXACT pt_maskratio partition)
            g = torch.Generator().manual_seed(PT_SEED)
            perm = torch.randperm(n_tok, generator=g)
            ratio_l1, ratio_hidden, ratio_err = [], [], []
            for r in PT_MASK_RATIOS:
                kk = max(1, int(round(n_tok * r)))
                pred_i = perm[:kk].sort().values
                enc_i = perm[kk:].sort().values
                l1_r, out_r, tgt_r = masked_predict_l1(
                    encoder, predictor, pixel, expand_mask(enc_i, 1), expand_mask(pred_i, 1))
                e = (out_r - tgt_r).abs().mean(-1)[0].float().cpu().numpy()
                ratio_l1.append(float(l1_r[0]))
                ratio_err.append(tokens_to_grid(e, pred_i.numpy(), num_frames))
                h2 = np.zeros(n_tok, bool); h2[pred_i.numpy()] = True
                ratio_hidden.append(h2.reshape(Tp, Hp, Wp))
            slope = float(perclip_slope(np.array(ratio_l1)[None, :], PT_MASK_RATIOS)[0])
            slope_ref = pt_maskratio.compute(encoder, predictor, batch[i:i + 1], num_frames)
            if not np.allclose(slope_ref[0], slope, rtol=0.05, atol=1e-4):
                sys.exit(f"FATAL[{label}]: demo maskratio slope {slope:.5f} != pt_maskratio {slope_ref[0]:.5f}")
            hero["D_l1"], hero["D_hidden"], hero["D_err"], hero["D_slope"] = \
                ratio_l1, ratio_hidden, ratio_err, slope

        pl["hero"][k] = hero
        del pixel
        torch.cuda.empty_cache()

    del encoder, predictor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[m14] {label} done · GPU freed ({torch.cuda.memory_allocated() // 2**20} MiB left)")
    return pl


# ── PIL compositor primitives ────────────────────────────────────────────────────

_FONT_PATH = font_manager.findfont("DejaVu Sans")
_FONT_BOLD = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))


def _font(sz, bold=True):
    return ImageFont.truetype(_FONT_BOLD if bold else _FONT_PATH, sz)


def _err_color(t):
    """t∈[0,1] → green(right) → yellow → red(wrong)."""
    t = float(np.clip(t, 0, 1))
    return (int(80 + 175 * t), int(200 - 140 * t), 45)


def _tile_img(frame: np.ndarray, hidden2d, mode, err2d=None, vmin=0.0, vmax=1.0, alpha=0.6, dim=0.85):
    """frame (crop,crop,3) uint8 → PIL with per-tile treatment.
    mode='input'  : hidden tiles multiplied toward black (what the model receives).
    mode='report' : hidden tiles tinted green→red by that tile's prediction error, colour-stretched
                    over the OBSERVED [vmin, vmax] across ALL models (self-eyeball 2026-07-13:
                    e/vmax landed every tile mid-orange for both models — the FROZEN-vs-OURS gap
                    was invisible; min-max stretch makes the better model visibly greener)."""
    img = frame.astype(np.float32).copy()
    crop = frame.shape[0]
    Hp, Wp = hidden2d.shape
    ts = crop // Hp
    span = max(vmax - vmin, 1e-9)
    for a in range(Hp):
        for b in range(Wp):
            if not hidden2d[a, b]:
                continue
            ys, xs = slice(a * ts, (a + 1) * ts), slice(b * ts, (b + 1) * ts)
            if mode == "input":
                img[ys, xs] *= (1.0 - dim)
            else:
                e = err2d[a, b] if err2d is not None else np.nan
                if np.isnan(e):
                    continue
                col = np.array(_err_color((e - vmin) / span), np.float32)
                img[ys, xs] = (1 - alpha) * img[ys, xs] + alpha * col
    return Image.fromarray(img.astype(np.uint8))


def _sparkline(draw, x0, y0, w, h, series, vmax, color):
    if not series:
        return
    bw = max(2, w // max(len(series), 1) - 2)
    for j, v in enumerate(series):
        bh = max(2, int(h * min(v / max(vmax, 1e-9), 1.0)))
        draw.rectangle([x0 + j * (bw + 2), y0 + h - bh, x0 + j * (bw + 2) + bw, y0 + h], fill=color)


class Canvas:
    """One 1280x720 dark frame: title bar, N square panels with captions, ticker BAND below the
    panels (VM20: a side ticker column shrank the panels to 255 px and left a ~37% dead band —
    the bottom band uses that space and the panels grow to ~300 px)."""

    def __init__(self, dcfg):
        self.W, self.H = dcfg["canvas_w"], dcfg["canvas_h"]
        self.pp = dcfg["panel_px"]

    def frame(self, title, subtitle, panels, tickers=None, footer=None, key=None, insets=None):
        """panels: [(PIL_square, caption, rgb)] · tickers: [dict(label,value_txt,sub,series,vmax,rgb)]
        · key: one-line colour legend repeated ON EVERY FRAME (VM11/C6: a key only on the title card
        leaves 22s of tinted panels unexplained) · insets: [(PIL, caption, rgb)] — the scene-W
        ROI zoom row, rendered in the bottom band (mutually exclusive with tickers)."""
        im = Image.new("RGB", (self.W, self.H), _BG)
        d = ImageDraw.Draw(im)
        d.text((self.W // 2, 26), title, font=_font(24), fill=_FG, anchor="mm")
        d.text((self.W // 2, 56), subtitle, font=_font(14), fill=_SUB, anchor="mm")
        n = len(panels)
        pw = min(self.pp, (self.W - 40 - 12 * (n - 1)) // max(n, 1))
        x = (self.W - (pw * n + 12 * (n - 1))) // 2
        y0 = 92
        for (p, cap, crgb) in panels:
            im.paste(p.resize((pw, pw), Image.LANCZOS), (x, y0))
            for li, line in enumerate(cap.split("\n")):
                d.text((x + pw // 2, y0 + pw + 16 + 18 * li), line, font=_font(13), fill=crgb, anchor="mm")
            x += pw + 12
        band = y0 + pw + 52                                   # bottom band: key + tickers OR insets
        if key:
            d.text((self.W // 2, band), key, font=_font(14), fill=_ACC, anchor="mm")
        if insets:
            ni = len(insets)
            iw = insets[0][0].width
            ix = (self.W - (iw * ni + 24 * (ni - 1))) // 2
            iy = band + 18
            for (ip, icap, irgb) in insets:
                im.paste(ip, (ix, iy))
                d.rectangle([ix, iy, ix + iw, iy + iw], outline=(255, 82, 82), width=2)
                d.text((ix + iw // 2, iy + iw + 12), icap, font=_font(12), fill=irgb, anchor="mm")
                ix += iw + 24
        if tickers:
            tw = (self.W - 60) // len(tickers)
            for j, t in enumerate(tickers):
                x0 = 30 + j * tw
                d.text((x0 + tw // 2, band + 34), f"{t['label']}    {t['value_txt']}",
                       font=_font(22), fill=t["rgb"], anchor="mm")
                d.text((x0 + tw // 2, band + 60), t.get("sub", ""), font=_font(12), fill=_SUB, anchor="mm")
                _sparkline(d, x0 + tw // 4, band + 74, tw // 2, 40, t["series"], t["vmax"], t["rgb"])
                d.text((x0 + tw // 2, band + 128), "error per hidden time-slot →", font=_font(11),
                       fill=_SUB, anchor="mm")
        if footer:
            d.text((self.W // 2, self.H - 20), footer, font=_font(13), fill=_ACC, anchor="mm")
        return im


class Sink:
    def __init__(self, frames_dir: Path):
        self.dir, self.i = frames_dir, 0
        frames_dir.mkdir(parents=True, exist_ok=True)
        # the RENDER phase is the longest and used to be a silent gap after "rendering frames …"
        # (user 2026-07-14: every runbook command must show a live pbar) — one frame = one tick.
        self.pbar = tqdm(desc="render", unit="frame")

    def add(self, img: Image.Image, repeat=1):
        img.save(self.dir / f"{self.i:06d}.png")
        first = self.dir / f"{self.i:06d}.png"
        self.i += 1
        for _ in range(repeat - 1):
            (self.dir / f"{self.i:06d}.png").write_bytes(first.read_bytes())
            self.i += 1
        self.pbar.update(repeat)

    def close(self):
        self.pbar.close()


def _pretty(key):
    """'walking_varanasi_00112_112150' → 'walking / varanasi · #112150' — the clip id STAYS
    (VM10: two clips labelled identically 'walking / varanasi' were indistinguishable)."""
    parts = key.split("_")
    return f"{parts[0]} / {parts[1]} · #{parts[-1]}"


def _slot_of(i, n_disp, Tp):
    return min(Tp - 1, int(i / max(n_disp, 1) * Tp))


# ── Scenes (every panel = the clip PLAYING) ─────────────────────────────────────

def scene_cards(cv, sink, dcfg, labels, n_clips, scenes, verdict_rows=None):
    hold = int(dcfg["title_card_s"] * dcfg["display_fps"])
    im = Image.new("RGB", (cv.W, cv.H), _BG)
    d = ImageDraw.Draw(im)
    if verdict_rows is None:
        if "W" in scenes:
            # VM19/C6: the intro must describe THIS scene's actual encodings (the old card promised
            # green/red report cards + numbers that scene W never shows).
            d.text((cv.W // 2, 180), "Who imagines the future better?", font=_font(36), fill=_FG, anchor="mm")
            d.text((cv.W // 2, 250), "   vs   ".join(labels), font=_font(26), fill=_ACC, anchor="mm")
            d.text((cv.W // 2, 320), "real DenseWorld street clips — the SECOND HALF of each clip is hidden from both models",
                   font=_font(16), fill=_SUB, anchor="mm")
            d.text((cv.W // 2, 388), "each middle panel PLAYS the real first half, then CUTS to that model's own imagined future",
                   font=_font(15), fill=_SUB, anchor="mm")
            d.text((cv.W // 2, 416), "watch the cut: one video keeps the street, one loses it  ·  white lines = the REAL future's outline",
                   font=_font(15), fill=_SUB, anchor="mm")
            d.text((cv.W // 2, 444), "red-bordered tiles at the bottom = the same heavy blur of all three videos — match FROZEN / OURS to the REAL one",
                   font=_font(15), fill=_SUB, anchor="mm")   # VM26: legend derived from CURRENT elements
            d.text((cv.W // 2, 512), _HONESTY, font=_font(13), fill=(96, 125, 139), anchor="mm")
        else:
            head = _METRIC_NAME["C"] if scenes == {"C"} else "the headline metrics"
            verb = "SEE" if head.endswith("metrics") else "SEES"                 # MIN1 grammar
            d.text((cv.W // 2, 180), f"How {head} {verb} a video", font=_font(36), fill=_FG, anchor="mm")
            d.text((cv.W // 2, 250), "   vs   ".join(labels), font=_font(26), fill=_ACC, anchor="mm")
            d.text((cv.W // 2, 320), "V-JEPA 2.1 ViT-g (1B) encoder + its own predictor  ·  real DenseWorld street clips",
                   font=_font(16), fill=_SUB, anchor="mm")
            d.text((cv.W // 2, 390), "the model predicts DESCRIPTIONS of hidden patches — never pixels —", font=_font(15), fill=_SUB, anchor="mm")
            d.text((cv.W // 2, 418), "so we paint its report card on the real frames:  green = it was right  ·  red = it was wrong",
                   font=_font(15), fill=_SUB, anchor="mm")
            d.text((cv.W // 2, 490), "every number = the paper eval suite's own code (utils.pt_* / predictor_eval)",
                   font=_font(13), fill=(96, 125, 139), anchor="mm")
    else:
        # VM20: vertically-centered block (no dead middle band) · VM12/VM14: CI spelled out + the
        # demo↔paper arm-name mapping stated on-card.
        tail = [
            ("the paper eval scores this SAME computation on 23,106 held-out clips (FULL 116k run) —", _SUB),
            ("there OURS beats FROZEN on causal future-block L1 by 87.8x the confidence interval", _ACC),
            ("of the difference — far beyond noise (forest_plot_frozen_ci)", _ACC),
            ("paper arm names:  OURS diheavy = surgical_3stage_DI_diheavy  ·  FROZEN 2.1 = frozen", (96, 125, 139)),
        ]
        n_lines = 1 + len(verdict_rows) + len(tail)
        y = (cv.H - n_lines * 40) // 2
        d.text((cv.W // 2, y), "Verdict", font=_font(30), fill=_FG, anchor="mm")
        y += 56
        for row, col in verdict_rows:
            d.text((cv.W // 2, y), row, font=_font(16), fill=col, anchor="mm")
            y += 40
        y += 12
        for row, col in tail:
            d.text((cv.W // 2, y), row, font=_font(14), fill=col, anchor="mm")
            y += 32
    sink.add(im, hold)


def scene_a(cv, sink, dcfg, models, disp, heroes, num_frames):
    Tp, _, _, _ = token_grid(num_frames)
    n_show = int(dcfg["scene_a_s"] * dcfg["display_fps"])
    for k in heroes:
        n_disp = disp[k].shape[0]
        pcas = {m["label"]: pca_rgb(m["feats_final"][k], num_frames) for m in models}
        for j in range(n_show):
            i = int(j / n_show * n_disp)
            s = _slot_of(i, n_disp, Tp)
            panels = [(Image.fromarray(disp[k][i]), f"Original · {_pretty(k)}", _FG)]
            for mi, m in enumerate(models):
                rgb = (pcas[m["label"]][s] * 255).astype(np.uint8)
                f = disp[k].shape[1] // rgb.shape[0]
                big = np.repeat(np.repeat(rgb, f, 0), f, 1)
                panels.append((Image.fromarray(big), m["label"], _MODEL_COLORS[mi]))
            sink.add(cv.frame("Scene A — what the model sees",
                              "same colour = tracked as the same thing across time (official V-JEPA dense-feature view)",
                              panels,
                              footer="every 2-frame 16x16 brick becomes a description-vector — shown here projected to colour"))


def _advantage_img(frame, hidden2d, err_a, err_b, alpha):
    """Single 'who wins WHERE' panel (user 2026-07-14: two mottled report cards looked identical):
    per hidden tile, GREEN = model B (OURS) predicted this patch better, RED = model A (FROZEN) did;
    tint intensity = how large the gap is (normalized by the max |gap|)."""
    img = frame.astype(np.float32).copy()
    crop = frame.shape[0]
    Hp, Wp = hidden2d.shape
    ts = crop // Hp
    gap = err_a - err_b                                    # >0 → B (OURS) better on this tile
    gmax = np.nanmax(np.abs(gap)) or 1e-9
    for a in range(Hp):
        for b in range(Wp):
            if not hidden2d[a, b] or np.isnan(gap[a, b]):
                continue
            ys, xs = slice(a * ts, (a + 1) * ts), slice(b * ts, (b + 1) * ts)
            t = float(np.clip(abs(gap[a, b]) / gmax, 0, 1))
            col = np.array((60, 190, 60) if gap[a, b] > 0 else (210, 60, 45), np.float32)
            aa = alpha * (0.25 + 0.75 * t)                 # faint = tiny gap, saturated = big gap
            img[ys, xs] = (1 - aa) * img[ys, xs] + aa * col
    return Image.fromarray(img.astype(np.uint8))


def scene_masked(cv, sink, dcfg, scene, models, disp, k, num_frames, subtitle, footer):
    """Scenes B & C — Original | model input | report card per model (+ advantage map when A/B)."""
    Tp, _, _, _ = token_grid(num_frames)
    n_disp = disp.shape[0]
    err_key, hid_key, val_key = f"{scene}_err", f"{scene}_hidden", f"{scene}_l1"
    vmax = max(np.nanmax(m["hero"][k][err_key]) for m in models)
    vmin = min(np.nanmin(m["hero"][k][err_key]) for m in models)
    series = {m["label"]: [] for m in models}
    seen = set()
    for i in range(n_disp):
        s = _slot_of(i, n_disp, Tp)
        if s not in seen:
            seen.add(s)
            for m in models:
                row = m["hero"][k][err_key][s]
                if not np.all(np.isnan(row)):
                    series[m["label"]].append(float(np.nanmean(row)))
        fr = disp[i]
        m0h = models[0]["hero"][k]
        panels = [(Image.fromarray(fr), f"Original · {_pretty(k)}", _FG),
                  (_tile_img(fr, m0h[hid_key][s], "input", dim=dcfg["mask_dim"]),
                   "model input\n(dim = hidden from the model)", _FG)]   # VM19: caption matches the render
        for mi, m in enumerate(models):
            h = m["hero"][k]
            panels.append((_tile_img(fr, h[hid_key][s], "report", h[err_key][s], vmin, vmax,
                                     dcfg["heat_alpha"], dcfg["mask_dim"]),
                           f"{m['label']}\nreport card", _MODEL_COLORS[mi]))
        key = ("report card:  green = described the hidden patch RIGHT  ·  red = WRONG  "
               "(colours stretched across both models)")
        if len(models) == 2:                               # A/B: ONE who-wins-WHERE panel (user 2026-07-14)
            hA, hB = models[0]["hero"][k], models[1]["hero"][k]
            a_short, b_short = models[0]["label"].split()[0], models[1]["label"].split()[0]
            panels.append((_advantage_img(fr, hA[hid_key][s], hA[err_key][s], hB[err_key][s],
                                          dcfg["heat_alpha"]),
                           f"who wins WHERE\ngreen={b_short} · red={a_short}", _ACC))
            key = (f"cards: green = right · red = wrong   |   map: green = {b_short} predicted that "
                   f"patch better · red = {a_short} did · saturation = gap")
        ticks = [{"label": m["label"], "value_txt": f"{m['hero'][k][val_key]:.3f}",
                  "sub": "hidden-half error · lower = better", "series": series[m["label"]], "vmax": vmax,
                  "rgb": _MODEL_COLORS[mi]} for mi, m in enumerate(models)]
        sink.add(cv.frame(f"Scene {scene} — {_METRIC_NAME[scene]}", subtitle, panels, ticks, footer,
                          key=key))


def scene_d(cv, sink, dcfg, models, disp, k, num_frames):
    Tp, _, _, _ = token_grid(num_frames)
    n_pass = int(dcfg["scene_d_pass_s"] * dcfg["display_fps"])
    n_disp = disp.shape[0]
    ratios = list(PT_MASK_RATIOS)
    vmax = max(np.nanmax(e) for m in models for e in m["hero"][k]["D_err"])
    vmin = min(np.nanmin(e) for m in models for e in m["hero"][k]["D_err"])
    l1max = max(v for m in models for v in m["hero"][k]["D_l1"])
    for j_r, r in enumerate(ratios):
        for j in range(n_pass):
            i = int(j / n_pass * n_disp)
            s = _slot_of(i, n_disp, Tp)
            fr = disp[i]
            m0h = models[0]["hero"][k]
            panels = [(Image.fromarray(fr), f"Original · {_pretty(k)}", _FG),
                      (_tile_img(fr, m0h["D_hidden"][j_r][s], "input", dim=dcfg["mask_dim"]),
                       f"model input\n{int(r * 100)}% hidden", _FG)]
            for mi, m in enumerate(models):
                h = m["hero"][k]
                panels.append((_tile_img(fr, h["D_hidden"][j_r][s], "report", h["D_err"][j_r][s],
                                         vmin, vmax, dcfg["heat_alpha"], dcfg["mask_dim"]),
                               f"{m['label']}\nerr {h['D_l1'][j_r]:.3f}", _MODEL_COLORS[mi]))
            ticks = [{"label": m["label"], "value_txt": f"{m['hero'][k]['D_slope']:+.4f}",
                      "sub": "slope = the metric", "series": m["hero"][k]["D_l1"][:j_r + 1],
                      "vmax": l1max, "rgb": _MODEL_COLORS[mi]} for mi, m in enumerate(models)]
            sink.add(cv.frame(f"Scene D — {_METRIC_NAME['D']}",
                              f"the SAME clip, 4 passes, more hidden each pass (eval sweep, seed {PT_SEED})",
                              panels, ticks,
                              "a jigsaw with more and more missing pieces — flatter error growth = better (lower slope wins)",
                              key="report card:  green = described the hidden patch RIGHT  ·  red = WRONG  "
                                  "(colours stretched across both models)"))


def _edge_mask(frame_uint8: np.ndarray, pctl: float) -> np.ndarray:
    """(H,W,3) uint8 → bool (H,W): Sobel-magnitude top-(100-pctl)% pixels, 1px dilated — the
    'real future' outline for the MIT Video-Diff style overlay (no cv2 in this venv → numpy)."""
    g = frame_uint8.astype(np.float32).mean(-1)
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1]))
    mag = gx + gy
    # no dilation: 1px lines — dilated 92nd-pctl edges buried the imagined panels under a white
    # snowstorm on crowded street scenes (self-eyeball 2026-07-14)
    return mag > np.percentile(mag, pctl)


def _overlay_edges(img: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = img.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * 255.0
    return out.astype(np.uint8)


def _contrast(frames: np.ndarray, c: float) -> np.ndarray:
    """Identical mild contrast stretch around each frame's mean (VM23 — presentation-only, applied
    to BOTH models' imagined streams with the same coefficient)."""
    x = frames.astype(np.float32)
    m = x.mean(axis=(1, 2, 3), keepdims=True)
    return np.clip(m + (x - m) * c, 0, 255).astype(np.uint8)


def decode_imaginations(models, decoders, k, num_frames, crop, patch, tube, w_contrast):
    """Each model's decoded 'imagined future' frames for clip k → {label: (half*tube, crop, crop, 3)}."""
    Tp, _, _, _ = token_grid(num_frames)
    half = Tp // 2
    imag = {}
    for m in models:
        z = torch.from_numpy(m["hero"][k]["W_pred"].astype(np.float32)).cuda()
        with torch.no_grad():
            px = decoders[m["label"]](z).cpu().numpy()
        imag[m["label"]] = _contrast(assemble_frames(px, Tp - half, crop, patch, tube), w_contrast)
    return imag


def select_decisive(models, decoders, disp, keys, num_frames, crop, patch, tube, dcfg):
    """VM22/VM23: rank clips by PIXEL-space decisiveness — mean|imagined_FROZEN − real_future| −
    mean|imagined_OURS − real_future| — and per clip pick the ROI tile-box where OURS beats FROZEN
    hardest in PIXELS (so the zoom inset shows FROZEN dissolving an object OURS keeps). Returns
    (ordered hero keys, {k: imag}, {k: roi_box_px}). FAIL LOUD if no clip favours OURS."""
    Tp, Hp, Wp, _ = token_grid(num_frames)
    ts = crop // Hp
    tiles = dcfg["inset_tiles"]
    ranked, imag_all, box_all = [], {}, {}
    for k in keys:
        imag = decode_imaginations(models, decoders, k, num_frames, crop, patch, tube, dcfg["w_contrast"])
        n_disp = disp[k].shape[0]
        midx = np.linspace(0, n_disp - 1, num_frames).round().astype(int)[num_frames // 2:]
        real = disp[k][midx].astype(np.float32)                       # the real future, model geometry
        la, lb = models[0]["label"], models[-1]["label"]
        errA = np.abs(imag[la].astype(np.float32) - real).mean(-1)    # (F, crop, crop)
        errB = np.abs(imag[lb].astype(np.float32) - real).mean(-1)
        gap_px = errA - errB                                          # >0 where OURS closer in pixels
        # per-tile decisiveness → ROI. VM24: weight the gap by the REAL tile's STRUCTURE (std) so
        # the crop lands on a salient patterned object — a flat single-hue object (blue bin) can be
        # "matched" by same-hue fog and visually invert the verdict.
        g = gap_px.mean(0)
        g_t = g.reshape(Hp, ts, Wp, ts).mean(axis=(1, 3))             # (Hp, Wp) per-tile mean gap
        r_std = real.std(0).mean(-1) if real.ndim == 4 else real.std(0)
        s_t = r_std.reshape(Hp, ts, Wp, ts).mean(axis=(1, 3))         # (Hp, Wp) real-structure weight
        w_t = g_t * (s_t / (s_t.mean() + 1e-6))
        sc = np.full((Hp - tiles + 1, Wp - tiles + 1), -1e9, np.float32)
        for y in range(sc.shape[0]):
            for x in range(sc.shape[1]):
                sc[y, x] = w_t[y:y + tiles, x:x + tiles].mean()
        by, bx = np.unravel_index(np.argmax(sc), sc.shape)
        box_all[k] = (bx * ts, by * ts, (bx + tiles) * ts, (by + tiles) * ts)
        imag_all[k] = imag
        ranked.append((float(gap_px.mean()), k))
        print(f"  [decisive] {k}: pixel gap {gap_px.mean():+.2f} (>{0} = OURS closer) · "
              f"ROI tile ({by},{bx}) score {sc[by, bx]:+.2f}")
    ranked.sort(reverse=True)
    if len(models) == 2 and ranked[0][0] <= 0:
        sys.exit("FATAL: no clip where OURS beats FROZEN in decoded-pixel space — the triptych "
                 "cannot honestly show a layman-visible win on this clip set; add more clips")
    heroes = [k for _g, k in ranked[: dcfg["n_hero_clips"]]]
    return heroes, imag_all, box_all


def scene_w(cv, sink, dcfg, models, disp, k, imag, box, num_frames):
    """Scene W — THE demo (plan_v2 v3 + VM23 layman comparators):
    `REAL past | <model>'s video (CONTINUITY CUT: plays the real past, then ITS imagination) | ... |
    REAL future`, sparse real-future outline on the imagined halves, red box + bottom ZOOM-INSET row
    on the PIXEL-decisive ROI. Pre-attentive judgments: 'whose video breaks at the cut' · 'whose
    picture fits the white outline' · 'which inset keeps the object alive'."""
    n_disp = disp.shape[0]
    n_past = n_disp // 2
    n_im = imag[models[0]["label"]].shape[0]
    ipx = dcfg["inset_px"]
    nb = dcfg["squint_blocks"]

    def _squint(fr):
        """VM25: identical aggressive downsample — the gestalt survives, fog cannot fake it."""
        return Image.fromarray(fr).resize((nb, nb), Image.BOX).resize((ipx, ipx), Image.NEAREST)

    for i in range(n_disp):
        second = i >= n_past
        j = min(int((i - n_past) / max(n_disp - n_past, 1) * n_im), n_im - 1) if second else 0
        p_past = Image.fromarray(disp[min(i, n_past - 1)])
        panels = [(p_past, "REAL past\n(all models saw this)", _FG)]
        insets = None
        if second:
            edges = _edge_mask(disp[i], dcfg["edge_pctl"])                     # real future outline
        for mi, m in enumerate(models):
            if second:
                fr_im = imag[m["label"]][j]
                pane = Image.fromarray(_overlay_edges(fr_im, edges, dcfg["edge_alpha"]))
                cap = f"{m['label']} — now IMAGINING\n(did its video just break?)"
            else:
                # CONTINUITY CUT (VM23): the model's panel PLAYS the real past at full brightness —
                # at the halfway cut it continues into its OWN imagination; the break is the verdict.
                pane = Image.fromarray(disp[i])
                cap = f"{m['label']} — playing the past\n(its imagination takes over at the cut)"
            panels.append((pane, cap, _MODEL_COLORS[mi]))
        if second:
            panels.append((Image.fromarray(disp[i]), "REAL future\n(the answer)", _ACC))
            # VM25 squint row — full-frame gestalt, NOT a crop (two ROI strategies both landed on fog)
            insets = [(_squint(imag[m["label"]][j]), f"{m['label']} (squint)", _MODEL_COLORS[mi])
                      for mi, m in enumerate(models)]
            insets.append((_squint(disp[i]), "REAL (squint)", _ACC))
        else:
            # VM20: the inset band is NOT left dead in the first half — a dimmed squint teaser holds it.
            tease = (disp[i].astype(np.float32) * 0.25).astype(np.uint8)
            insets = [(_squint(tease), "squint test appears\nwhen the future hides", _SUB)]
            panels.append((Image.fromarray((disp[i].astype(np.float32) * 0.02).astype(np.uint8)),
                           "REAL future\n(not shown yet)", _SUB))
        sink.add(cv.frame(f"{_METRIC_NAME['W']}   ·   {_pretty(k)}",
                          "the second half is HIDDEN from both models — each must imagine it from the first half alone",
                          panels,
                          key=("which video survived the cut?  ·  SQUINT TEST below: blur everything the "
                               "same way — which imagination still matches the REAL future?" if second else
                               "watch both middle panels at the cut — one keeps the scene, one loses it"),
                          footer=_HONESTY, insets=insets))


def scene_e(cv, sink, dcfg, models, disp, keys, num_frames):
    """4 clips playing in a grid; link-line thickness = pooled-feature cosine (first model)."""
    n_show = int(dcfg["scene_e_s"] * dcfg["display_fps"])
    m = models[0]
    F = np.stack([m["pooled"][k] for k in keys])
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    C = F @ F.T
    types = [k.split("_")[0] for k in keys]
    same = [C[a, b] for a in range(len(keys)) for b in range(len(keys)) if a < b and types[a] == types[b]]
    cross = [C[a, b] for a in range(len(keys)) for b in range(len(keys)) if a < b and types[a] != types[b]]
    margin = float(np.mean(same) - np.mean(cross))
    pw = 250
    pos = [(190, 110), (190, 420), (720, 110), (720, 420)]
    order = sorted(range(len(keys)), key=lambda a: types[a])
    for j in range(n_show):
        im = Image.new("RGB", (cv.W, cv.H), _BG)
        d = ImageDraw.Draw(im)
        d.text((cv.W // 2, 26), f"Scene E — {_METRIC_NAME['E']} · {m['label']}", font=_font(24), fill=_FG, anchor="mm")
        d.text((cv.W // 2, 56), "4 clips play together — line thickness = how ALIKE the model finds them "
                                "(walking-walking should beat walking-driving)", font=_font(14), fill=_SUB, anchor="mm")
        centers = {a: (pos[slot][0] + pw // 2, pos[slot][1] + pw // 2) for slot, a in enumerate(order)}
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                wpx = max(1, int(1 + 14 * max(C[a, b], 0) ** 2))
                col = (46, 125, 50) if types[a] == types[b] else (84, 110, 122)
                d.line([centers[a], centers[b]], fill=col, width=wpx)
        for slot, a in enumerate(order):
            k = keys[a]
            i = int(j / n_show * disp[k].shape[0])
            x, y = pos[slot]
            im.paste(Image.fromarray(disp[k][i]).resize((pw, pw), Image.LANCZOS), (x, y))
            d.text((x + pw // 2, y + pw + 14), _pretty(k), font=_font(13), fill=_FG, anchor="mm")
        d.text((cv.W - 120, 300), f"{margin:+.3f}", font=_font(30), fill=_ACC, anchor="mm")
        d.text((cv.W - 120, 350), "separation margin\nsame - cross\n(higher = better)", font=_font(12),
               fill=_SUB, anchor="mm")
        sink.add(im)
    return margin, C


# ── Contact sheet ────────────────────────────────────────────────────────────────

def contact_sheet(frames_dir: Path, out_png: Path, tiles: int):
    pngs = sorted(frames_dir.glob("*.png"))
    idx = np.linspace(0, len(pngs) - 1, tiles).round().astype(int)
    ims = [Image.open(pngs[i]).resize((480, 270)) for i in idx]
    side = int(np.ceil(np.sqrt(tiles)))
    sheet = Image.new("RGB", (side * 480, side * 270), "black")
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % side) * 480, (k // side) * 270))
    sheet.save(out_png)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="m14 v2 — video-editing demo of the headline metrics")
    p.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH",
                   help="repeatable: 'FROZEN 2.1=checkpoints/….pt' (1 = solo, 2 = A/B)")
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--demo-config", type=Path, required=True)
    p.add_argument("--clips-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scenes", required=True,
                   help="'all' or a subset like 'W' / 'BC' — option-2 workflow: perfect ONE scene first")
    p.add_argument("--decoder", action="append", default=None, metavar="LABEL=PATH",
                   help="scene W: per-model m15 decoder.pt — LABEL must match a --ckpt label")
    p.add_argument("--sanity-one-clip", action="store_true",
                   help="E4 smoke: first clip only — fast end-to-end check")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("FATAL: CUDA required (1B encoder+predictor forwards) — no CPU fallback")
    scenes = set("ABCDEW") if args.scenes.lower() == "all" else set(args.scenes.upper())
    if not scenes <= set("ABCDEW"):
        sys.exit(f"FATAL: --scenes must be 'all' or a subset of ABCDEW, got '{args.scenes}'")
    specs = []
    for spec in args.ckpt:
        if "=" not in spec:
            sys.exit(f"FATAL: --ckpt must be LABEL=PATH, got '{spec}'")
        label, path = spec.split("=", 1)
        if not Path(path).exists():
            sys.exit(f"FATAL: ckpt not found: {path}")
        specs.append((label, Path(path)))
    if len(specs) > 2:
        sys.exit("FATAL: at most 2 --ckpt (A/B) supported")

    dcfg = yaml.safe_load(args.demo_config.read_text())["demo"]
    num_frames = get_pipeline_config()["probe"]["num_frames"]        # metric parity with m12d/m12e
    _mc = yaml.safe_load(args.model_config.read_text())["model"]
    crop, patch, tube = _mc["crop_size"], _mc["patch_size"], _mc["tubelet_size"]
    decoders = {}
    if "W" in scenes:                                    # scene W needs one m15 decoder PER model
        for spec in (args.decoder or []):
            if "=" not in spec:
                sys.exit(f"FATAL: --decoder must be LABEL=PATH, got '{spec}'")
            label, path = spec.split("=", 1)
            if not Path(path).exists():
                sys.exit(f"FATAL: decoder not found: {path} (run m15 --stage train first)")
            decoders[label] = load_token_decoder(Path(path))
        missing = [lbl for lbl, _ in specs if lbl not in decoders]
        if missing:
            sys.exit(f"FATAL: scene W needs --decoder for every model; missing: {missing}")
    out = args.output_dir
    frames_dir = out / f"frames_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(args.clips_dir.glob("*.mp4"))
    if not mp4s:
        sys.exit(f"FATAL: no .mp4 in {args.clips_dir}")
    if args.sanity_one_clip:
        mp4s = mp4s[:1]
    print(f"[m14 v2] {len(mp4s)} clips · scenes={''.join(sorted(scenes))} · num_frames={num_frames} · "
          f"crop={crop} · models: {[s[0] for s in specs]} · display {dcfg['display_fps']} fps")

    disp, batch_rows, keys = {}, [], []
    for mp4 in tqdm(mp4s, desc="decode"):
        allf = decode_all_frames(mp4)
        model_idx = np.linspace(0, allf.shape[0] - 1, num_frames).round().astype(int)
        batch_rows.append(resize_and_normalize(allf[model_idx], crop))
        disp[mp4.stem] = resize_center_crop_uint8(allf, crop)
        keys.append(mp4.stem)
    batch = torch.stack(batch_rows)

    if "W" in scenes:
        hero_pool = keys      # VM22: decisiveness selection needs EVERY clip's payload — pick later
    else:
        hero_pool = []
        for k in keys:                                   # prefer one clip per tour_type …
            if k.split("_")[0] not in [h.split("_")[0] for h in hero_pool]:
                hero_pool.append(k)
        for k in keys:                                   # … then fill with same-type clips (a crowd
            if len(hero_pool) >= dcfg["n_hero_clips"]:   # set may be ALL 'walking' — 2026-07-14)
                break
            if k not in hero_pool:
                hero_pool.append(k)
        hero_pool = hero_pool[: dcfg["n_hero_clips"]]
    print(f"[m14 v2] hero pool: {hero_pool}")

    mask_gen = build_mask_gen(num_frames)
    shared_masks = {}
    if "B" in scenes:
        for k in hero_pool:
            me, mp_ = mask_gen(1)
            shared_masks[k] = ((torch.stack(me, 0) if isinstance(me, list) else me).to("cuda"),
                               (torch.stack(mp_, 0) if isinstance(mp_, list) else mp_).to("cuda"))

    models = [compute_payload(label, ck, args.model_config, batch, keys, hero_pool, num_frames,
                              shared_masks, scenes) for label, ck in specs]

    if "W" in scenes:   # VM22: heroes + ROI chosen by PIXEL-space decisiveness AFTER decoding
        heroes, imag_all, box_all = select_decisive(models, decoders, disp, hero_pool,
                                                    num_frames, crop, patch, tube, dcfg)
        print(f"[m14 v2] decisive heroes: {heroes}")
    else:
        heroes = hero_pool

    # ── render ──
    cv = Canvas(dcfg)
    sink = Sink(frames_dir)
    print("[m14 v2] rendering frames …")
    scene_cards(cv, sink, dcfg, [m["label"] for m in models], len(keys), scenes)
    if "A" in scenes:
        scene_a(cv, sink, dcfg, models, disp, heroes, num_frames)
    for k in heroes:
        if "W" in scenes:
            scene_w(cv, sink, dcfg, models, disp[k], k, imag_all[k], box_all[k], num_frames)
        if "B" in scenes:
            scene_masked(cv, sink, dcfg, "B", models, disp[k], k, num_frames,
                         "random spacetime blocks hidden (the m12d eval mask) — the model must describe them",
                         "cover parts of the flip-book — green tiles it described right, red it got wrong")
        if "C" in scenes:
            scene_masked(cv, sink, dcfg, "C", models, disp[k], k, num_frames,
                         "the ENTIRE second half is hidden — predict the future from the past only (the pt_causal eval mask)",
                         "hide the second half of the movie — who guesses the future better from the first half?")
        if "D" in scenes:
            scene_d(cv, sink, dcfg, models, disp[k], k, num_frames)
    margin, C = (scene_e(cv, sink, dcfg, models, disp, keys, num_frames)
                 if "E" in scenes else (None, np.zeros((len(keys), len(keys)))))   # None, never NaN (VM21)

    # verdict rows — FAIL LOUD on any missing value: a sign-off card must never ship an em-dash
    # placeholder where a number belongs (VM21).
    rows = []
    for k in heroes:
        vals = {}
        for m in models:
            h = m["hero"][k]
            if {"C", "W"} & scenes:
                if "C_l1" not in h or not np.isfinite(h["C_l1"]):
                    sys.exit(f"FATAL (VM21): verdict value C_l1 missing for {m['label']} on {k}")
                vals[m["label"]] = h["C_l1"]
        for mi, m in enumerate(models):
            bits = []
            h = m["hero"][k]
            if {"C", "W"} & scenes:
                mark = "   <- wins (lower error)" if (len(models) > 1 and
                                                      m["label"] == min(vals, key=vals.get)) else ""
                bits.append(f"{_METRIC_NAME['C']}: {vals[m['label']]:.4f}{mark}")
            if "B" in scenes:
                bits.append(f"{_METRIC_NAME['B']}: {h['B_l1']:.4f}")
            if "D" in scenes:
                bits.append(f"{_METRIC_NAME['D']}: {h['D_slope']:+.4f}")
            if not bits:
                sys.exit(f"FATAL (VM21): no verdict values for {m['label']} on {k} (scenes={scenes})")
            rows.append((f"{_pretty(k)} · {m['label']}   —   " + "   ·   ".join(bits),
                         (236, 239, 241) if mi == 0 else _MODEL_COLORS[1]))
    if margin is not None:
        rows.append((f"{_METRIC_NAME['E']} (all clips): {margin:+.3f}  (higher = better)", _ACC))
    scene_cards(cv, sink, dcfg, [m["label"] for m in models], len(keys), scenes, verdict_rows=rows)

    sink.close()
    mp4_path = out / ("demo_sanity.mp4" if args.sanity_one_clip else f"demo_{''.join(sorted(scenes))}.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(dcfg["display_fps"]),
                    "-i", str(frames_dir / "%06d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", str(dcfg["crf"]), str(mp4_path)], check=True)
    contact_sheet(frames_dir, out / "contact_sheet.png", dcfg["sheet_tiles"])
    mj = {"num_frames": num_frames, "scenes": "".join(sorted(scenes)),
          "mask_ratios": list(PT_MASK_RATIOS), "models": {}}
    for m in models:
        mj["models"][m["label"]] = {
            "per_clip": {k: {kk: vv for kk, vv in m["hero"][k].items()
                             if kk in ("B_l1", "C_l1", "D_slope", "D_l1")} for k in heroes},
            "motion_cos_margin": margin,
        }
    (out / "demo_metrics.json").write_text(json.dumps(mj, indent=1))
    print(f"[m14 v2] DONE → {mp4_path} ({sink.i} frames @ {dcfg['display_fps']} fps = "
          f"{sink.i / dcfg['display_fps']:.0f}s) · contact_sheet.png · demo_metrics.json")


if __name__ == "__main__":
    main()
