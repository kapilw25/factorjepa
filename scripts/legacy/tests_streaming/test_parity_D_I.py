"""D_I streaming parity test — bitwise equality vs legacy m11 path.

Picks 5 sample clips with has_D_I=True, runs each clip through both paths:
  - Path A (legacy): manually execute m11 _process_one_clip(regen_di_only=True)
    body — np.load(mask.npz) → decode_video_bytes → filter blacklist →
    make_interaction_tubes_from_bboxes/_centroids.
  - Path B (streaming): call stream_interaction_tubes() from utils/factor_streaming.

Asserts: len(tubes_A) == len(tubes_B) AND np.array_equal(tubes_A[i], tubes_B[i])
for every tube. Any mismatch fails with clip_key + tube_idx + diff stats.

Why this matters: ONLY test that guarantees the wrapper added no bugs. Blocks
Phase 1 wiring of training.py D_I streaming branch.

    python -u scripts/tests_streaming/test_parity_D_I.py 2>&1 | tee logs/parity_D_I.log
"""
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from m11_factor_datasets import (   # noqa: E402
    make_interaction_tubes_from_bboxes,
    make_interaction_tubes_from_centroids,
)
from utils.factor_streaming import stream_interaction_tubes   # noqa: E402
from utils.video_io import decode_video_bytes   # noqa: E402


N_SAMPLES = 5
SUBSET_JSON = PROJECT_ROOT / "data/subset_10k.json"
LOCAL_DATA = PROJECT_ROOT / "data/subset_10k_local"
MANIFEST = PROJECT_ROOT / "outputs/full/m11_factor_datasets/factor_manifest.json"
MASKS_DIR = PROJECT_ROOT / "outputs/full/m10_sam_segment/masks"
TRAIN_YAML = PROJECT_ROOT / "configs/train/ch11_surgery_v15c.yaml"


def _build_tar_index(local_dir: Path) -> dict:
    """Scan TAR shards → {clip_key: (tar_path, base)}."""
    index = {}
    for tar_path in sorted(local_dir.glob("*.tar")):
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".json"):
                    base = member.name[:-len(".json")]
                    try:
                        meta = json.loads(tar.extractfile(member).read())
                        clip_key = f"{meta['section']}/{meta['video_id']}/{meta['source_file']}"
                        index[clip_key] = (str(tar_path), base)
                    except (json.JSONDecodeError, KeyError):
                        continue
    return index


def _legacy_path(mp4_bytes: bytes, mask_npz: Path, interaction_cfg: dict,
                 num_frames: int, tmp_dir: str, clip_key: str) -> list:
    """Exact copy of m11 _process_one_clip regen_di_only branch (lines 646-691)
    — so we compare stream_interaction_tubes against the SAME computation path.
    If this diverges from m11, the parity test is invalid; update accordingly.
    """
    data = np.load(mask_npz, allow_pickle=True)
    interactions = json.loads(str(data["interactions_json"])) if "interactions_json" in data else []
    centroids = json.loads(str(data["centroids_json"])) if "centroids_json" in data else {}
    per_object_bboxes = json.loads(str(data["per_object_bboxes_json"])) if "per_object_bboxes_json" in data else {}
    obj_id_to_cat = json.loads(str(data["obj_id_to_cat_json"])) if "obj_id_to_cat_json" in data else {}

    frames_tensor = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames=num_frames)
    if frames_tensor is None:
        return []
    frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
    frames_np = ((frames_np * 255).astype(np.uint8)
                 if frames_np.max() <= 1.0 else frames_np.astype(np.uint8))

    tubes: list = []
    if interactions and interaction_cfg["enabled"]:
        blacklist = {tuple(sorted(pair)) for pair in interaction_cfg["category_pair_blacklist"]}
        filtered: list = []
        for ev in interactions:
            ca = ev.get("cat_a") or obj_id_to_cat.get(str(ev["obj_a"]))
            cb = ev.get("cat_b") or obj_id_to_cat.get(str(ev["obj_b"]))
            if ca is not None and cb is not None and tuple(sorted((ca, cb))) in blacklist:
                continue
            ev = dict(ev)
            ev["cat_a"], ev["cat_b"] = ca, cb
            filtered.append(ev)
        margin = interaction_cfg["tube_margin_pct"]
        if per_object_bboxes:
            tubes = make_interaction_tubes_from_bboxes(frames_np, filtered, per_object_bboxes, margin)
        elif centroids:
            tubes = make_interaction_tubes_from_centroids(frames_np, filtered, centroids, margin)
    return tubes


def main():
    # Load interaction_cfg from v15c yaml (interaction_mining.enabled=true)
    with open(TRAIN_YAML) as f:
        interaction_cfg = yaml.safe_load(f)["interaction_mining"]
    if not interaction_cfg.get("enabled"):
        print(f"FATAL: {TRAIN_YAML} has interaction_mining.enabled=false; "
              "use the v15c yaml for parity (D_I must be enabled).")
        sys.exit(1)

    # Pick 5 clips that have n_interactions > 0 (direct check in m10 segments.json)
    segs_file = PROJECT_ROOT / "outputs/full/m10_sam_segment/segments.json"
    segs = json.load(open(segs_file))
    candidates = [k for k, v in segs.items() if v.get("n_interactions", 0) > 0]
    if len(candidates) < N_SAMPLES:
        print(f"FATAL: only {len(candidates)} clips have n_interactions>0; need ≥{N_SAMPLES}")
        sys.exit(1)
    # Use deterministic order: first N_SAMPLES by clip_key sort
    candidates.sort()
    sample_keys = candidates[:N_SAMPLES]

    print(f"Building TAR index from {LOCAL_DATA}...")
    tar_index = _build_tar_index(LOCAL_DATA)
    sample_keys = [k for k in sample_keys if k in tar_index][:N_SAMPLES]
    if len(sample_keys) < N_SAMPLES:
        print(f"FATAL: only {len(sample_keys)}/{N_SAMPLES} sample clips present in TAR index")
        sys.exit(1)
    print(f"  TAR index built: {len(tar_index)} total clips")
    print(f"  Sampled {len(sample_keys)} clips for parity test")

    total_tubes = 0
    total_mismatches = 0
    failed_clips = []

    for i, clip_key in enumerate(sample_keys):
        tar_path, base = tar_index[clip_key]
        with tarfile.open(tar_path, "r") as tar:
            mp4_bytes = tar.extractfile(tar.getmember(f"{base}.mp4")).read()

        safe_key = clip_key.replace("/", "__")
        mask_npz = MASKS_DIR / f"{safe_key}.npz"
        if not mask_npz.exists():
            print(f"  [{i+1}/{N_SAMPLES}] SKIP {clip_key}: mask.npz missing")
            continue

        with tempfile.TemporaryDirectory(prefix="parity_di_") as tmp:
            tubes_legacy = _legacy_path(mp4_bytes, mask_npz, interaction_cfg,
                                        num_frames=16, tmp_dir=tmp, clip_key=clip_key)
            tubes_stream = stream_interaction_tubes(
                mp4_bytes=mp4_bytes, mask_npz_path=mask_npz,
                interaction_cfg=interaction_cfg, num_frames=16,
                tmp_dir=tmp, clip_key=clip_key,
            )

        print(f"  [{i+1}/{N_SAMPLES}] {clip_key}: "
              f"legacy={len(tubes_legacy)} tubes · streaming={len(tubes_stream)} tubes")

        if len(tubes_legacy) != len(tubes_stream):
            print(f"    ❌ tube COUNT mismatch")
            failed_clips.append((clip_key, "count_mismatch"))
            total_mismatches += 1
            continue

        clip_mismatches = 0
        for j, (t_leg, t_str) in enumerate(zip(tubes_legacy, tubes_stream)):
            if t_leg.shape != t_str.shape:
                print(f"    ❌ tube {j} SHAPE mismatch: legacy={t_leg.shape} · streaming={t_str.shape}")
                clip_mismatches += 1
            elif not np.array_equal(t_leg, t_str):
                diff = np.abs(t_leg.astype(int) - t_str.astype(int))
                print(f"    ❌ tube {j} BYTES differ: shape={t_leg.shape} "
                      f"max_diff={diff.max()} mean_diff={diff.mean():.3f}")
                clip_mismatches += 1
            total_tubes += 1
        if clip_mismatches == 0:
            print(f"    ✅ all {len(tubes_legacy)} tubes bitwise-identical")
        else:
            failed_clips.append((clip_key, f"{clip_mismatches} tube mismatches"))
            total_mismatches += clip_mismatches

    print()
    print(f"{'='*60}")
    print(f"D_I PARITY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Clips tested: {len(sample_keys)}")
    print(f"  Total tubes compared: {total_tubes}")
    print(f"  Mismatches: {total_mismatches}")
    if failed_clips:
        print(f"  Failed clips:")
        for ck, reason in failed_clips:
            print(f"    {ck}: {reason}")
    if total_mismatches == 0 and total_tubes > 0:
        print(f"  ✅ ALL GREEN — streaming bitwise-matches legacy m11 path")
        sys.exit(0)
    print(f"  ❌ FAILED — fix stream_interaction_tubes before wiring training.py")
    sys.exit(1)


if __name__ == "__main__":
    main()
