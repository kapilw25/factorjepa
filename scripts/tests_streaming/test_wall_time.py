"""Tier-3 wall-time regression guard — projects streaming vs legacy throughput.

At the hot-path level, streaming costs ~2.4 s/clip (scipy σ=15 gaussian blur)
vs legacy ~50 ms/clip (np.load disk read). This is by design: streaming trades
single-call latency for ~340 GB disk savings at 10K. The key question isn't
single-call latency — it's whether DataLoader prefetch can keep num_workers=16
flowing fast enough that the GPU never stalls on V-JEPA 2.1 ViT-G's ~55 s/step.

This test:
  1. Measures single-call latency of stream_factor() + load_factor_clip()
     on 10 iter8 1K POC clips (mean of 10 runs each).
  2. Projects 10K × 1 epoch × BS=32 wall time at num_workers ∈ {0, 4, 16}.
  3. Asserts that projected streaming wall at num_workers=16 ≤ 10 h
     (matches ch11_surgery.yaml FULL budget assumption).
  4. Emits metrics JSON to outputs/sanity/factor_streaming_regression.json
     for CI archiving.

Fails if:
  - stream_factor single-call latency > 10 s (scipy regression)
  - projected 10K wall at num_workers=16 > 10 h (worse than 1K POC extrapolation)

Run:
    python -u scripts/tests_streaming/test_wall_time.py 2>&1 | tee logs/streaming_walltime.log
"""
import json
import random
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from utils.data_download import _get_clip_key
from utils.factor_streaming import stream_factor
from utils.training import load_factor_clip


POC_DIR = REPO / "outputs/poc/m11_factor_datasets"
MASKS_DIR = REPO / "outputs/poc/m10_sam_segment/masks"
LOCAL_DATA = REPO / "data/val_1k_local"
TRAIN_CFG = REPO / "configs/train/ch11_surgery.yaml"
OUT_JSON = REPO / "outputs/sanity/factor_streaming_regression.json"

N_CLIPS = 10
SEED = 42

# 10K × 1 epoch × BS=32 → 312 steps. GPU wall/step on 96 GB Blackwell ≈ 55 s.
STEPS_10K = 312
GPU_STEP_WALL_S = 55.0
BUDGET_HOURS = 10.0


def build_mp4_index(local_data: Path) -> dict:
    mp4_index = {}
    for tar_path in sorted(local_data.glob("*.tar")):
        with tarfile.open(tar_path, "r") as tar:
            entries = {}
            for m in tar.getmembers():
                base = m.name.rsplit(".", 1)[0]
                ext = m.name.rsplit(".", 1)[-1] if "." in m.name else ""
                entries.setdefault(base, {})[ext] = m
            for base, parts in entries.items():
                if "json" not in parts or "mp4" not in parts:
                    continue
                jb = tar.extractfile(parts["json"]).read()
                k = _get_clip_key(jb)
                if k and k not in mp4_index:
                    mp4_index[k] = (str(tar_path), base)
    return mp4_index


def fetch_mp4(mp4_index, clip_key):
    tp, base = mp4_index[clip_key]
    with tarfile.open(tp, "r") as tar:
        return tar.extractfile(tar.getmember(f"{base}.mp4")).read()


def project_wall(single_call_s: float, num_workers: int) -> float:
    """Project total training wall (seconds) for 10K × 1 epoch."""
    clips_per_step = 32
    if num_workers == 0:
        cpu_wall_per_step = single_call_s * clips_per_step
    else:
        cpu_wall_per_step = (single_call_s * clips_per_step) / num_workers
    # Prefetch hides CPU behind GPU; step wall is max(CPU, GPU).
    step_wall = max(cpu_wall_per_step, GPU_STEP_WALL_S)
    return step_wall * STEPS_10K


def main() -> int:
    print("🧪 Tier-3 wall-time regression guard\n")
    cfg_yaml = yaml.safe_load(open(TRAIN_CFG))["factor_datasets"]
    factor_cfg = {
        "layout_method": cfg_yaml["layout_patch_method"],
        "agent_method":  cfg_yaml["agent_patch_method"],
        "matte_factor":  cfg_yaml["soft_matte_factor"],
        "blur_sigma":    cfg_yaml["blur_sigma"],
        "feather_sigma": cfg_yaml["feather_sigma"],
    }

    manifest = json.loads((POC_DIR / "factor_manifest.json").read_text())
    candidates = []
    for clip_key, info in manifest.items():
        if not (info.get("has_D_L") and info.get("has_D_A")):
            continue
        safe = clip_key.replace("/", "__")
        dl_npy = POC_DIR / "D_L" / f"{safe}.npy"
        mask_npz = MASKS_DIR / f"{safe}.npz"
        if dl_npy.exists() and mask_npz.exists():
            candidates.append((clip_key, dl_npy, mask_npz))
    picked = random.Random(SEED).sample(candidates, N_CLIPS)
    mp4_index = build_mp4_index(LOCAL_DATA)

    print(f"📊 Measuring single-call latencies on {N_CLIPS} clips...\n")
    legacy_times, stream_times = [], []
    with tempfile.TemporaryDirectory(prefix="wt_") as tmp:
        for i, (clip_key, dl_npy, mask_npz) in enumerate(picked, 1):
            # Legacy: np.load(.npy) → normalize (cached disk I/O)
            t0 = time.perf_counter()
            _ = load_factor_clip(dl_npy, num_frames=16, crop_size=384)
            legacy_times.append(time.perf_counter() - t0)

            # Streaming: MP4 decode + mask align + scipy gaussian σ=15 + normalize
            mp4_bytes = fetch_mp4(mp4_index, clip_key)
            t0 = time.perf_counter()
            _ = stream_factor(
                mp4_bytes=mp4_bytes, mask_npz_path=mask_npz,
                factor_type="D_L", factor_cfg=factor_cfg,
                num_frames=16, tmp_dir=tmp, clip_key=clip_key,
            )
            stream_times.append(time.perf_counter() - t0)
            print(f"  [{i:2d}/{N_CLIPS}] legacy={legacy_times[-1]*1000:6.1f}ms  "
                  f"stream={stream_times[-1]*1000:7.1f}ms")

    legacy_mean = float(np.mean(legacy_times))
    stream_mean = float(np.mean(stream_times))
    ratio = stream_mean / legacy_mean

    print(f"\n📈 Single-call means (per clip):")
    print(f"   legacy   = {legacy_mean*1000:7.1f} ms  (np.load .npy + normalize)")
    print(f"   stream   = {stream_mean*1000:7.1f} ms  (mp4 decode + scipy σ=15 + normalize)")
    print(f"   ratio    = {ratio:.1f}×  (streaming is slower per call, BY DESIGN)\n")

    # Project 10K training wall at different num_workers
    print(f"🚀 Projected 10K × 1 epoch × BS=32 wall (GPU step = {GPU_STEP_WALL_S}s):")
    projections = {}
    for nw in (0, 4, 16):
        wall_s = project_wall(stream_mean, nw)
        hours = wall_s / 3600.0
        step_wall = wall_s / STEPS_10K
        projections[f"num_workers_{nw}"] = {
            "step_wall_s": step_wall,
            "total_wall_h": hours,
        }
        tag = "✅" if hours <= BUDGET_HOURS else "❌"
        print(f"   {tag} num_workers={nw:2d}: step≈{step_wall:5.1f}s, total≈{hours:5.2f}h")

    legacy_projection = project_wall(legacy_mean, 0)
    print(f"\n   (legacy single-process projection: {legacy_projection/3600:.2f}h — "
          f"disk-only baseline, ignored for GPU-bound runs)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "single_call_s": {"legacy_mean": legacy_mean, "stream_mean": stream_mean,
                          "stream_vs_legacy_ratio": ratio},
        "projections_10k_1ep_bs32": projections,
        "gpu_step_wall_s": GPU_STEP_WALL_S,
        "budget_hours": BUDGET_HOURS,
    }, indent=2))
    print(f"\n💾 Metrics saved: {OUT_JSON}")

    # Gate 1: single-call regression
    failed = False
    if stream_mean > 10.0:
        print(f"\n❌ FAIL: stream_factor single-call {stream_mean:.1f}s > 10s ceiling "
              f"— scipy regressed?")
        failed = True
    # Gate 2: num_workers=16 projection fits budget
    proj_16h = projections["num_workers_16"]["total_wall_h"]
    if proj_16h > BUDGET_HOURS:
        print(f"\n❌ FAIL: projected 10K wall at num_workers=16 = {proj_16h:.2f}h "
              f"exceeds budget {BUDGET_HOURS}h")
        failed = True

    if failed:
        return 1
    print(f"\n✅ Tier-3 passed — projected 10K wall at num_workers=16 "
          f"({proj_16h:.2f}h) within budget ({BUDGET_HOURS}h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
