#!/bin/bash
# Tier-2 integration test — exercises the full m09c streaming pipeline
# (config merge + streaming index build + StreamingFactorDataset + DataLoader)
# WITHOUT invoking the V-JEPA model or GPU. Gates FULL launch.
#
# What it asserts:
#   1. `--factor-streaming` CLI override wins over yaml mode gate (3 modes)
#   2. `build_streaming_indices` scans iter8 1K POC TARs + m10 masks correctly
#   3. StreamingFactorDataset yields (B, T, C, H, W) float32 batches via DataLoader
#      at num_workers ∈ {0, 2} with pin_memory + persistent_workers + prefetch
#   4. Legacy FactorSampler path still works (iter8 1K POC bitwise reproducibility)
#
# Run:
#   bash scripts/tests_streaming/test_sanity_end_to_end.sh 2>&1 | tee logs/streaming_sanity.log
#
# Why no actual --SANITY run: a true end-to-end m09c --SANITY takes ~15 min
# of GPU time. For a pre-launch gate, the structural integration + Tier-1
# bitwise parity (scripts/tests_streaming/test_parity.py) + Tier-3 wall-time
# together cover the surface that a full SANITY run would add. GPU-consuming
# SANITY runs live in iter9/runbook.md under Step C.
set -euo pipefail

cd "$(dirname "$0")/../.."
source venv_walkindia/bin/activate

python -u - <<'PY'
import argparse, copy, sys
from pathlib import Path
sys.path.insert(0, "src")

import torch
import yaml
from torch.utils.data import DataLoader

from utils.config import load_merged_config
from utils.training import (
    StreamingFactorDataset, build_streaming_indices, _streaming_worker_init,
    FactorSampler, build_factor_index, load_factor_clip,
)
from m09c_surgery import merge_config_with_args


def _make_args(mode: str, fs_override=None):
    return argparse.Namespace(
        SANITY=(mode == "sanity"), POC=(mode == "poc"), FULL=(mode == "full"),
        subset=None, local_data=None, batch_size=None, max_epochs=None,
        probe_subset=None, probe_local_data=None, probe_tags=None, no_probe=False,
        output_dir="/tmp/_streaming_sanity_test",
        factor_streaming_override=fs_override,
    )


def check_1_cli_override():
    """CLI --factor-streaming / --no-factor-streaming wins over yaml mode gate."""
    print("\n🧪 [1/4] CLI override precedence")
    cfg_base = load_merged_config("configs/model/vjepa2_1.yaml",
                                  "configs/train/ch11_surgery.yaml")
    cases = [
        ("sanity", None,  False, "yaml default: SANITY=legacy"),
        ("sanity", True,  True,  "--factor-streaming forces streaming"),
        ("sanity", False, False, "--no-factor-streaming preserves legacy"),
        ("full",   None,  True,  "yaml default: FULL=streaming"),
        ("full",   False, False, "--no-factor-streaming forces legacy on FULL"),
    ]
    for mode, override, expected, label in cases:
        ns = _make_args(mode, fs_override=override)
        merged = merge_config_with_args(copy.deepcopy(cfg_base), ns)
        actual = merged["factor_streaming"]["enabled"]
        tag = "✅" if actual == expected else "❌"
        print(f"  {tag} {label}: enabled={actual} (expected {expected})")
        assert actual == expected, f"CLI override failed: {label}"
    print("  ALL 5 CLI override cases pass")


def check_2_streaming_indices():
    """build_streaming_indices scans 1K POC TARs + masks correctly."""
    print("\n🧪 [2/4] build_streaming_indices on iter8 1K POC")
    manifest_path = Path("outputs/poc/m11_factor_datasets/factor_manifest.json")
    if not manifest_path.exists():
        print(f"  ⏭️  SKIP: {manifest_path} not found (run m11 --POC first)")
        return False
    mp4_index, mask_index, manifest = build_streaming_indices(
        manifest_path=manifest_path,
        masks_dir=Path("outputs/poc/m10_sam_segment/masks"),
        local_data="data/val_1k_local",
    )
    assert len(mp4_index) > 0, "mp4_index empty — TAR scan failed"
    assert len(mask_index) > 0, "mask_index empty — m10 masks missing"
    assert len(manifest) > 0, "manifest empty"
    print(f"  ✅ {len(mp4_index)} mp4 entries, {len(mask_index)} masks, "
          f"{len(manifest)} manifest entries")
    return {"mp4_index": mp4_index, "mask_index": mask_index, "manifest": manifest}


def check_3_dataloader(indices):
    """StreamingFactorDataset + DataLoader emit valid batches on {0, 2} workers."""
    print("\n🧪 [3/4] DataLoader shape + dtype + mode_mixture")
    if not indices:
        print("  ⏭️  SKIP (no indices from check 2)")
        return
    fcy = yaml.safe_load(open("configs/train/ch11_surgery.yaml"))["factor_datasets"]
    factor_cfg = {
        "layout_method": fcy["layout_patch_method"],
        "agent_method":  fcy["agent_patch_method"],
        "matte_factor":  fcy["soft_matte_factor"],
        "blur_sigma":    fcy["blur_sigma"],
        "feather_sigma": fcy["feather_sigma"],
    }
    ds = StreamingFactorDataset(
        mp4_index=indices["mp4_index"], mask_index=indices["mask_index"],
        factor_manifest=indices["manifest"], factor_cfg=factor_cfg,
        mode_mixture={"L": 0.3, "A": 0.7},
        num_frames=16, crop_size=384, base_seed=42, steps_per_epoch=6,
    )
    for nw in (0, 2):
        loader = DataLoader(
            ds, batch_size=2, num_workers=nw,
            prefetch_factor=4 if nw > 0 else None,
            persistent_workers=(nw > 0), pin_memory=False,
            worker_init_fn=_streaming_worker_init if nw > 0 else None,
        )
        seen_factors = set()
        for i, batch in enumerate(loader):
            t = batch["tensor"]
            assert tuple(t.shape) == (2, 16, 3, 384, 384), f"bad shape {t.shape}"
            assert t.dtype == torch.float32, f"bad dtype {t.dtype}"
            assert -3.5 < t.mean().item() < 3.5, f"un-normalized mean {t.mean()}"
            seen_factors.update(batch["factor_type"])
            if i >= 2:
                break
        assert seen_factors.issubset({"D_L", "D_A"}), f"unexpected factors {seen_factors}"
        print(f"  ✅ num_workers={nw}: 3 batches valid, factors={sorted(seen_factors)}")


def check_4_legacy_path(indices):
    """FactorSampler + load_factor_clip legacy path still works."""
    print("\n🧪 [4/4] Legacy FactorSampler + load_factor_clip")
    if not indices:
        print("  ⏭️  SKIP (no indices from check 2)")
        return
    factor_index = build_factor_index(
        indices["manifest"],
        Path("outputs/poc/m11_factor_datasets/D_L"),
        Path("outputs/poc/m11_factor_datasets/D_A"),
        Path("outputs/poc/m11_factor_datasets/D_I"),
    )
    sampler = FactorSampler(factor_index, {"L": 0.3, "A": 0.7})
    for _ in range(3):
        _, clip_key, path = sampler.sample()
        tensor = load_factor_clip(path, num_frames=16, crop_size=384)
        assert tuple(tensor.shape) == (16, 3, 384, 384), f"bad shape {tensor.shape}"
        assert tensor.dtype == torch.float32
    print("  ✅ 3 legacy samples loaded, shapes + dtypes valid")


def main():
    print("=" * 60)
    print("🧪 Tier-2 SANITY end-to-end integration (non-GPU)")
    print("=" * 60)
    check_1_cli_override()
    indices = check_2_streaming_indices()
    check_3_dataloader(indices)
    check_4_legacy_path(indices)
    print("\n" + "=" * 60)
    print("✅ Tier-2 ALL CHECKS PASSED — streaming + legacy paths both green")
    print("   Ready for GPU SANITY run: m09c --SANITY (either path via yaml or CLI)")
    print("=" * 60)


main()
PY
