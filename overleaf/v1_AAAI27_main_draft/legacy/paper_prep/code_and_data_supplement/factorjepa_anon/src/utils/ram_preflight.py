"""Static host-RAM preflight for the training scheduler — the RAM sibling of ngpu_run's disk/cpu
preflights. Catches DATA-SCALED anon footprints that OOM the cgroup BEFORE any GPU spend.

WHY (iter19 2026-07-04): m09a1 preloads the whole val split into HOST RAM ("Collecting N val clips
into memory"). At FULL the 5% val split is ~5,750 clips ≈ 40 G anon → OOM-thrash / SIGKILL on a
~127 G cgroup. SANITY (232 val) is 25× smaller so it NEVER surfaced the ceiling — a resource-scaling
bug no small-data smoke can catch, because SANITY scales DOWN the data and thus the footprint. This
computes the val-preload upper bound (bounded by validation.max_val_clips after the cap fix) from
config and checks it against the cgroup cap, so an oversized max_val_clips / tiny box FATALs at
preflight instead of 3 minutes into a 19 h run.

Per-clip host bytes = num_frames × crop² × 3 (uint8 decoded frames) — ALL config-derived (base
train cfg data.num_frames + validation.max_val_clips; model cfg model.crop_size), no hardcodes.
The empirical iter19 footprint (~7 MB/clip: 16 × 384² × 3) matches this formula.

Self-test (CPU):
    python -u src/utils/ram_preflight.py configs/train/base_optimization.yaml configs/model/vjepa2_1_vitg.yaml
"""
import sys

import yaml

from utils.cgroup_monitor import read_cgroup_memory_anon, read_cgroup_memory_limit

_GIB = 1024 ** 3
# Canonical config entry points (same treatment as ngpu_run's literal "configs/pipeline.yaml" /
# arm_registry's "configs/arm_registry.yaml") — these are the pipeline's fixed roots, not data paths.
_BASE_TRAIN_CFG = "configs/train/base_optimization.yaml"
_PIPELINE_CFG   = "configs/pipeline.yaml"


def val_preload_gb(max_val_clips: int, num_frames: int, crop: int) -> float:
    """Upper-bound host RAM (GiB) held by the in-memory val preload: `max_val_clips` decoded uint8
    clips of shape (num_frames, 3, crop, crop). This is the data-scaled anon term that OOM'd iter19."""
    per_clip_bytes = num_frames * crop * crop * 3          # 3 channels, uint8 decoded frames
    return max_val_clips * per_clip_bytes / _GIB


def estimate(train_cfg_path: str, model_cfg_path: str) -> dict:
    """Config-derived val-preload footprint vs the live cgroup RAM cap. FAIL-LOUD on any missing key
    (cfg[...] with no .get default) so a config drift crashes here, not mid-run."""
    tcfg = yaml.safe_load(open(train_cfg_path))
    mcfg = yaml.safe_load(open(model_cfg_path))
    num_frames = tcfg["data"]["num_frames"]
    max_val    = tcfg["validation"]["max_val_clips"]
    crop       = mcfg["model"]["crop_size"]
    val_gb     = val_preload_gb(max_val, num_frames, crop)
    cap_bytes  = read_cgroup_memory_limit()
    anon_bytes = read_cgroup_memory_anon()
    return {
        "num_frames": num_frames,
        "crop": crop,
        "max_val_clips": max_val,
        "per_clip_mb": num_frames * crop * crop * 3 / 1e6,
        "val_preload_gb": val_gb,
        "cap_gb": (cap_bytes / _GIB) if cap_bytes else None,
        "anon_now_gb": (anon_bytes / _GIB) if anon_bytes else None,
    }


def estimate_for_backbone(backbone: str,
                          train_cfg_path: str = _BASE_TRAIN_CFG,
                          pipeline_cfg_path: str = _PIPELINE_CFG) -> dict:
    """Resolve the backbone's model config via pipeline.yaml backbone_model_configs, then estimate().
    KeyError (fail-loud) if the backbone is not registered."""
    pcfg = yaml.safe_load(open(pipeline_cfg_path))
    model_cfg_path = pcfg["backbone_model_configs"][backbone]
    return estimate(train_cfg_path, model_cfg_path)


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -u src/utils/ram_preflight.py <train_cfg.yaml> <model_cfg.yaml>")
    r = estimate(sys.argv[1], sys.argv[2])
    cap = f"{r['cap_gb']:.0f}G" if r["cap_gb"] else "unlimited (no cgroup)"
    pct = f"{100 * r['val_preload_gb'] / r['cap_gb']:.0f}% of cap" if r["cap_gb"] else "n/a"
    print("[ram-preflight selftest]")
    print(f"  per-clip     = {r['num_frames']}f × {r['crop']}² × 3 = {r['per_clip_mb']:.1f} MB (uint8)")
    print(f"  val preload  = {r['max_val_clips']} clips × {r['per_clip_mb']:.1f} MB = {r['val_preload_gb']:.1f} G  ({pct})")
    print(f"  cgroup cap   = {cap} · anon now = {r['anon_now_gb'] or 0:.1f} G")
