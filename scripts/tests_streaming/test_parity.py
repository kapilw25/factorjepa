"""Bitwise parity: stream_factor(raw_mp4, mask) == m11's disk-written .npy.

This is the PRIMARY CORRECTNESS GATE for the streaming refactor. If this
fails, the iter9 m09c streaming path will produce different training tensors
than iter8 legacy path → invalidates the Surgery vs Frozen comparison on the
scaling curve.

Picks 5 random clips from outputs/poc/m11_factor_datasets/ (iter8 1K POC
artifacts) that have BOTH D_L and D_A .npy + a matching m10 mask.npz + an
MP4 in data/val_1k_local/subset-*.tar. For each clip × factor, asserts
np.array_equal(legacy_npy, stream_factor_output).

Run:
    source venv_walkindia/bin/activate
    python -u scripts/tests_streaming/test_parity.py 2>&1 | tee logs/streaming_parity.log

Exits 0 on success, 1 on any mismatch. 10 assertions total (5 clips × 2 factors).
"""
import json
import random
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from utils.data_download import _get_clip_key
from utils.factor_streaming import stream_factor


POC_DIR = REPO / "outputs/poc/m11_factor_datasets"
MASKS_DIR = REPO / "outputs/poc/m10_sam_segment/masks"
LOCAL_DATA = REPO / "data/val_1k_local"
TRAIN_CFG = REPO / "configs/train/ch11_surgery.yaml"
N_CLIPS = 5
SEED = 42


def build_mp4_index(local_data: Path) -> dict:
    """Scan TARs, return {clip_key: (tar_path, base_name)}."""
    mp4_index = {}
    for tar_path in sorted(local_data.glob("*.tar")):
        with tarfile.open(tar_path, "r") as tar:
            entries = {}
            for member in tar.getmembers():
                base = member.name.rsplit(".", 1)[0]
                ext = member.name.rsplit(".", 1)[-1] if "." in member.name else ""
                entries.setdefault(base, {})[ext] = member
            for base, parts in entries.items():
                if "json" not in parts or "mp4" not in parts:
                    continue
                json_bytes = tar.extractfile(parts["json"]).read()
                key = _get_clip_key(json_bytes)
                if key and key not in mp4_index:
                    mp4_index[key] = (str(tar_path), base)
    return mp4_index


def fetch_mp4_bytes(mp4_index: dict, clip_key: str) -> bytes:
    tar_path, base = mp4_index[clip_key]
    with tarfile.open(tar_path, "r") as tar:
        return tar.extractfile(tar.getmember(f"{base}.mp4")).read()


def main() -> int:
    print(f"🧪 Tier-1 parity test: stream_factor vs m11 disk .npy")
    print(f"   POC dir: {POC_DIR}")
    print(f"   masks:   {MASKS_DIR}")
    print(f"   data:    {LOCAL_DATA}")

    with open(TRAIN_CFG) as f:
        train_cfg = yaml.safe_load(f)
    factor_cfg_yaml = train_cfg["factor_datasets"]
    factor_cfg = {
        "layout_method": factor_cfg_yaml["layout_patch_method"],
        "agent_method":  factor_cfg_yaml["agent_patch_method"],
        "matte_factor":  factor_cfg_yaml["soft_matte_factor"],
        "blur_sigma":    factor_cfg_yaml["blur_sigma"],
        "feather_sigma": factor_cfg_yaml["feather_sigma"],
    }

    manifest = json.loads((POC_DIR / "factor_manifest.json").read_text())

    dl_dir = POC_DIR / "D_L"
    da_dir = POC_DIR / "D_A"
    candidates = []
    for clip_key, info in manifest.items():
        if not (info.get("has_D_L") and info.get("has_D_A")):
            continue
        safe_key = clip_key.replace("/", "__")
        dl_npy = dl_dir / f"{safe_key}.npy"
        da_npy = da_dir / f"{safe_key}.npy"
        mask_npz = MASKS_DIR / f"{safe_key}.npz"
        if dl_npy.exists() and da_npy.exists() and mask_npz.exists():
            candidates.append((clip_key, dl_npy, da_npy, mask_npz))
    if len(candidates) < N_CLIPS:
        print(f"❌ Not enough candidates with both D_L+D_A+mask present "
              f"(have {len(candidates)}, need {N_CLIPS})")
        return 1

    rng = random.Random(SEED)
    picked = rng.sample(candidates, N_CLIPS)

    print(f"\n📦 Building MP4 index from {LOCAL_DATA} ...")
    mp4_index = build_mp4_index(LOCAL_DATA)
    print(f"   {len(mp4_index)} clips indexed")

    n_pass = 0
    n_fail = 0
    with tempfile.TemporaryDirectory(prefix="parity_") as tmp:
        for i, (clip_key, dl_npy, da_npy, mask_npz) in enumerate(picked, 1):
            if clip_key not in mp4_index:
                print(f"[{i}/{N_CLIPS}] ⚠️  {clip_key}: not in MP4 index, skip")
                n_fail += 1
                continue
            mp4_bytes = fetch_mp4_bytes(mp4_index, clip_key)

            for factor_type, npy_path in [("D_L", dl_npy), ("D_A", da_npy)]:
                legacy = np.load(npy_path)
                stream = stream_factor(
                    mp4_bytes=mp4_bytes,
                    mask_npz_path=mask_npz,
                    factor_type=factor_type,
                    factor_cfg=factor_cfg,
                    num_frames=16,
                    tmp_dir=tmp,
                    clip_key=clip_key,
                )
                ok = np.array_equal(legacy, stream)
                tag = "✅" if ok else "❌"
                info = (f"shape={legacy.shape} dtype={legacy.dtype} "
                        f"mean={legacy.mean():.2f}")
                print(f"[{i}/{N_CLIPS}] {tag} {clip_key} {factor_type}: {info}")
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                    diff = (legacy.astype(np.int32) - stream.astype(np.int32))
                    nmis = int((diff != 0).sum())
                    print(f"    mismatch pixels: {nmis}/{legacy.size} "
                          f"({100*nmis/legacy.size:.3f}%)  |  max |diff|={int(np.abs(diff).max())}")

    print(f"\n{'='*60}")
    total = n_pass + n_fail
    print(f"RESULT: {n_pass}/{total} passed, {n_fail}/{total} failed")
    if n_fail == 0:
        print("✅ BITWISE PARITY CONFIRMED — streaming path matches m11 disk output")
        return 0
    print("❌ PARITY BROKEN — do NOT proceed to m09c wire-in (file #4)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
