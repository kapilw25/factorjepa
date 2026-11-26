"""Shared helpers for the iter20 visual demo family (m14 renderer + m15 pixel decoder).

Lives in utils/ because BOTH m14_metric_demo.py and m15_pixel_decoder.py need it and m*.py
cross-imports are banned (src/CLAUDE.md rule 32 / reusable-helpers-in-utils).

Decode substitution (documented in both callers' docstrings): PyAV is absent in the demo venv
(venv_denseworld) — frames come from the system ffmpeg CLI; sampling semantics match
utils.video_io's uniform indexing, preprocessing is the EXACT eval recipe
(utils.frozen_features.resize_and_normalize).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from utils.frozen_features import IMAGENET_MEAN, IMAGENET_STD, resize_and_normalize


def ffprobe_wh_fps(mp4: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(mp4)],
        capture_output=True, text=True, check=True)
    st = json.loads(out.stdout)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return int(st["width"]), int(st["height"]), float(num) / float(den)


def decode_all_frames(mp4: Path) -> np.ndarray:
    """(N, H, W, 3) uint8 — EVERY frame via an ffmpeg rawvideo pipe (display path)."""
    W, H, _fps = ffprobe_wh_fps(mp4)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    n = len(raw) // (W * H * 3)
    if n == 0:
        sys.exit(f"FATAL: 0 frames decoded from {mp4}")
    return np.frombuffer(raw, np.uint8)[: n * W * H * 3].reshape(n, H, W, 3).copy()


def probe_n_frames(mp4: Path) -> int:
    """Container frame count (mp4 header — no decode). FATAL if the container doesn't carry it."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "json", str(mp4)],
        capture_output=True, text=True, check=True)
    nb = json.loads(out.stdout)["streams"][0].get("nb_frames")
    if not nb or not str(nb).isdigit():
        sys.exit(f"FATAL: container has no nb_frames: {mp4} — cannot select-decode identically")
    return int(nb)


def decode_frames_select(mp4: Path, indices) -> np.ndarray:
    """(len(indices), H, W, 3) uint8 — decode ONLY the requested frame indices via ffmpeg's select
    filter. PIXEL-IDENTICAL to decode_all_frames(mp4)[indices] (same decoder; verified
    np.array_equal on real denseworld clips, /gpu-bottleneck 2026-07-14) while piping ~20 MB instead
    of ~350 MB: 1565 ms → 516 ms measured. FATAL if the decoded count mismatches (container lied)."""
    W, H, _fps = ffprobe_wh_fps(mp4)
    sel = "+".join(f"eq(n\\,{int(i)})" for i in indices)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"select='{sel}'", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    got = len(raw) // (W * H * 3)
    if got != len(list(indices)):
        sys.exit(f"FATAL: select-decode returned {got} frames, wanted {len(list(indices))}: {mp4}")
    return np.frombuffer(raw, np.uint8).reshape(got, H, W, 3).copy()


def resize_center_crop_uint8(frames: np.ndarray, crop: int) -> np.ndarray:
    """(N,H,W,3) uint8 → (N,crop,crop,3) uint8 — SAME geometry as the eval recipe (reuses
    resize_and_normalize then de-normalizes, so display pixels == model-input pixels)."""
    t = resize_and_normalize(frames, crop)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (t * std + mean).clamp(0, 1)
    return (x.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


def tubelet_targets(norm_clip: torch.Tensor, num_frames: int, crop: int, patch: int, tube: int):
    """(T,3,crop,crop) normalized fp32 → (n_tok, tube*patch*patch*3) fp32 in [0,1] pixel space,
    token order = t*(Hp*Wp) + h*Wp + w (matches temporal_token_idx's row-major layout)."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (norm_clip * std + mean).clamp(0, 1)
    Tp, Hp, Wp = num_frames // tube, crop // patch, crop // patch
    x = x.reshape(Tp, tube, 3, Hp, patch, Wp, patch)
    return x.permute(0, 3, 5, 1, 4, 6, 2).reshape(Tp * Hp * Wp, tube * patch * patch * 3)


def assemble_frames(tok_pixels: np.ndarray, n_slots: int, crop: int, patch: int, tube: int):
    """(n_slots*Hp*Wp, tube*patch*patch*3) in [0,1] → (n_slots*tube, crop, crop, 3) uint8 —
    inverse of tubelet_targets; works on ANY contiguous run of temporal slots (e.g. just the
    hidden future half)."""
    Hp, Wp = crop // patch, crop // patch
    x = torch.from_numpy(tok_pixels).reshape(n_slots, Hp, Wp, tube, patch, patch, 3)
    x = x.permute(0, 3, 6, 1, 4, 2, 5).reshape(n_slots * tube, 3, Hp * patch, Wp * patch)
    return (x.permute(0, 2, 3, 1).clamp(0, 1).numpy() * 255).astype(np.uint8)


class TokenDecoder(torch.nn.Module):
    """m15 tubelet-inversion head: final-layer token (in_dim) → its tubelet's pixels (out_dim)."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, out_dim), torch.nn.Sigmoid())

    def forward(self, z):
        return self.net(z)


def load_token_decoder(decoder_pt: Path) -> TokenDecoder:
    ck = torch.load(decoder_pt, map_location="cuda", weights_only=False)
    dec = TokenDecoder(ck["in_dim"], ck["hidden"], ck["out_dim"]).cuda()
    dec.load_state_dict(ck["state_dict"])
    return dec.eval()
