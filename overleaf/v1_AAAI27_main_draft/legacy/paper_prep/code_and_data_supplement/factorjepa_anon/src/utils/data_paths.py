"""Canonical local-data path accessors (corpus manifest + m10/m11 output dirs). GPU-free.
    python -c "from utils.data_paths import factor_manifest_path; print(factor_manifest_path('data/full_local'))"
"""
# SINGLE SOURCE for the on-disk path conventions that m10/m11 write and m09* read.
# Every subdir/filename is read from configs/pipeline.yaml data.* (NOT hardcoded here)
# so the convention lives in exactly one place. Replaces the scattered literals that
# caused the iter15-class divergence (m09c1 read factor_manifest from --factor-dir,
# m09c2 reconstructed it from local_data + "m11_factor_datasets"). See CLAUDE.md
# "SHARED DERIVATION VIA CLI". FAIL LOUD via strict cfg[...] lookups (no .get defaults).
from pathlib import Path

from utils.config import get_pipeline_config


def _data_cfg() -> dict:
    return get_pipeline_config()["data"]


def artifact(key: str) -> str:
    """Canonical artifact basename/glob/suffix from pipeline.yaml `artifacts:`.

    iter18 W4 (2026-06-06): single source for every filename one module writes
    and another reads. Strict lookup — unknown key = KeyError = fail loud.
    Usage: out_dir / artifact("test_metrics")."""
    return get_pipeline_config()["artifacts"][key]


def corpus_manifest_path(local_data) -> Path:
    """The master/corpus manifest (all clips). Filename from pipeline.yaml
    data.master_manifest_name — same source run_train.sh uses for MASTER_MANIFEST."""
    return Path(local_data) / _data_cfg()["master_manifest_name"]


def factor_dir(local_data) -> Path:
    """m11_factor_datasets output dir (holds D_L/D_A/D_I + factor_manifest.json)."""
    return Path(local_data) / _data_cfg()["factor_subdir"]


def factor_manifest_path(local_data) -> Path:
    """factor_manifest.json inside the m11 factor dir."""
    return factor_dir(local_data) / _data_cfg()["factor_manifest_name"]


def factor_manifest_in(factor_dir_path) -> Path:
    """factor_manifest.json inside an EXPLICIT factor dir (e.g. m09c1's --factor-dir),
    so the manifest filename convention stays single-sourced even when the dir is
    passed via CLI."""
    return Path(factor_dir_path) / _data_cfg()["factor_manifest_name"]


def masks_dir(local_data) -> Path:
    """m10 SAM mask dir: <local_data>/<masks_subdir>/masks (inner name from yaml)."""
    return Path(local_data) / _data_cfg()["masks_subdir"] / _data_cfg()["masks_inner_dirname"]


def tube_cache_dir(local_data) -> Path:
    """D_I interaction-tube read-through cache dir: <local_data>/<tube_cache_subdir>/.

    iter18 (2026-06-15): one .npz per (clip_key, cfg-hash) memoizing
    stream_interaction_tubes()'s decode+mining output. Sibling of m10/m11 so it
    rides the same upload/download/clean conventions. Subdir name from
    pipeline.yaml data.tube_cache_subdir (single source — never hardcoded)."""
    return Path(local_data) / _data_cfg()["tube_cache_subdir"]


def shards_dir(local_data) -> Path:
    """m00d video-shard dir: <local_data>/<shard_subdir>/ (holds subset-*.tar).

    iter17 (2026-05-27): m00d now writes WebDataset shards into this subdir so
    every module owns a m<NN>_<name>/ dir (was the local_data ROOT — the one
    flat exception). Readers use find_video_shards / video_shard_glob below,
    which fall back to the root layout for corpora uploaded before this move."""
    return Path(local_data) / _data_cfg()["shard_subdir"]


def find_video_shards(local_data) -> list:
    """Sorted list of m00d video shards (subset-*.tar) for `local_data`.

    Prefers the canonical <shard_subdir>/ layout; falls back to the legacy
    root layout so corpora still in the old on-HF layout keep streaming.
    Name-anchored to `subset-*.tar` so it NEVER grabs masks-*.tar / D_*-*.tar
    (those live in sibling m10/m11 subdirs). May return [] (callers preserve
    the prior empty-glob behaviour: downstream FATALs on 0 clips)."""
    g = artifact("video_shard_glob")
    sub = sorted(shards_dir(local_data).glob(g))
    return sub if sub else sorted(Path(local_data).glob(g))


def video_shard_glob(local_data) -> str:
    """Glob STRING for HF datasets.load_dataset(data_files=...). Mirrors
    find_video_shards' subdir-then-root resolution. FAIL LOUD if neither
    layout has shards (load_dataset would otherwise yield a silent empty
    stream / cryptic error)."""
    g = artifact("video_shard_glob")
    sub = shards_dir(local_data)
    if any(sub.glob(g)):
        return str(sub / g)
    if any(Path(local_data).glob(g)):
        return str(Path(local_data) / g)
    raise FileNotFoundError(
        f"no {g} under {sub}/ or {local_data}/ — run "
        f"src/m00d_download_subset.py or hf_outputs.py download-data first")
