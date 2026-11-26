"""TAR-shard utility for HF 10k-file limit. CPU-only.

Huggingface dataset repos enforce a soft cap of ~10,000 files per repo. m10 alone
produces ~9300 .npz mask files for eval_10k → already at the limit. This utility
packs many small files into N evenly-sized .tar shards (mirrors the
`data/eval_10k_local/subset-{00000..00009}.tar` pattern that already works).

USAGE:
    # Pack: 9300 .npz files → 10 shards of ~930 files each
    python -u src/utils/tar_shard.py pack \\
        --input-dir data/eval_10k_local/m10_sam_segment/masks \\
        --shard-template "data/eval_10k_local/m10_sam_segment/masks-{shard:05d}.tar" \\
        --n-shards 10 --keep-source 2>&1 | tee logs/tar_shard_pack.log

    # Unpack: 10 shards → restored .npz directory
    python -u src/utils/tar_shard.py unpack \\
        --shards-glob "data/eval_10k_local/m10_sam_segment/masks-*.tar" \\
        --output-dir data/eval_10k_local/m10_sam_segment/masks \\
        2>&1 | tee logs/tar_shard_unpack.log

DESIGN:
- Idempotent pack: if all N shards already exist with non-zero size, skips work
  unless --force is passed. Skips files already INSIDE a shard via tarfile member
  enumeration (avoids double-add on resumed packs).
- Streaming write: tarfile.add(arcname=...) does NOT load the whole file into
  RAM, suitable for masks/ at multi-GB scale.
- Deterministic shard assignment: hash(filename) % n_shards → same file always
  lands in the same shard across runs (helps incremental upload).
- No GPU dependency. Used by hf_outputs.upload_data (pre-upload) +
  hf_outputs.download_data (post-download).
"""
import argparse
import glob
import hashlib
import sys
import tarfile
import time
from pathlib import Path


def _stable_shard_index(filename: str, n_shards: int) -> int:
    """Deterministic shard assignment via SHA1(filename) mod n_shards."""
    h = hashlib.sha1(filename.encode("utf-8")).hexdigest()
    return int(h, 16) % n_shards


def _shard_path(template: str, shard_idx: int) -> Path:
    """Render shard path from template like 'masks-{shard:05d}.tar'."""
    return Path(template.format(shard=shard_idx))


def pack_dir_to_shards(
    input_dir: Path,
    shard_template: str,
    max_shard_size_gb: float,
    keep_source: bool = True,
    force: bool = False,
) -> dict:
    """Pack all files in input_dir into TAR shards, rolling over at size cap.

    iter13 v13 FIX-24 (2026-05-07): redesigned to m00d-style size-driven
    streaming. Files are sorted (deterministic order) then streamed into a
    growing shard; when adding the next file would push the shard over
    `max_shard_size_gb`, the current shard closes and a new one opens.
    Naturally auto-scales 10K → 115K → larger without ANY n_shards retuning
    by the caller. Drops the prior hash-based bucketing approach (which
    produced uneven shard sizes — some 9 GB, some 50 MB at 115K).

    Idempotency: if any shard matching `shard_template` already exists and
    `force=False`, skip the entire pack (caller is expected to wipe via
    cache-policy=2 before re-packing).

    Args:
      input_dir: directory of leaf files to pack (e.g., masks/, D_L/, D_A/, D_I/)
      shard_template: format string with {shard:05d} placeholder
      max_shard_size_gb: soft cap per shard. A single file larger than this
                         FATALs (cannot fit in any shard). HF's fast-upload
                         sweet spot is ~1 GB; pipeline.yaml data.max_tar_shard_gb
                         is the canonical value.
      keep_source: leave .npz/.npy on disk after packing (default True since
                   m11 + m09c read individual files at runtime)
      force: repack even if shards already exist

    Returns: {n_files_packed, n_shards_written, total_bytes, elapsed_sec,
              max_shard_gb_observed, max_shard_size_gb}.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        print(f"FATAL: input_dir not a directory: {input_dir}")
        sys.exit(1)

    # Deterministic order: sorted by name → reproducible shard contents.
    files = sorted(p for p in input_dir.iterdir() if p.is_file())
    if not files:
        print(f"  [tar_shard pack] no files under {input_dir} — skipping")
        return {"n_files_packed": 0, "n_shards_written": 0, "total_bytes": 0,
                "elapsed_sec": 0.0, "max_shard_gb_observed": 0.0,
                "max_shard_size_gb": max_shard_size_gb}

    max_bytes = int(max_shard_size_gb * 1e9)

    # Oversized single files (> the soft cap) are UNSPLITTABLE by tar, so each lands ALONE in its
    # own dedicated shard that legitimately exceeds the cap. The roll-before-add loop below already
    # guarantees this: a non-empty shard always rolls before a too-big file is added (so the big
    # file starts a fresh shard), and the big file then rolls out the next file too (so it stays
    # alone). iter18 2026-06-08: this REPLACED a hard FATAL/sys.exit that aborted upload-full
    # mid-pack on the 30 GB cassle/ewc m09c_ckpt_latest.pt resume anchors — a backup tool must
    # never refuse to back a large file up; it shards it solo and logs it loudly.
    oversize = [f for f in files if f.stat().st_size > max_bytes]
    if oversize:
        print(f"  [tar_shard pack] {len(oversize)} file(s) exceed the {max_shard_size_gb:.1f} GB "
              f"cap → each gets its OWN over-cap shard (tar can't split one file): "
              f"{', '.join(f'{f.name}={f.stat().st_size / 1e9:.1f}GB' for f in oversize)}")

    # Idempotency: skip if any shard already exists (caller wipes via
    # cache-policy=2 before re-packing). Discovers existing shards by globbing
    # the template's parent dir for matching prefix.
    sample_path = _shard_path(shard_template, 0)
    if not force:
        # Use the {shard:NNNNN} prefix from sample_path's stem to glob siblings.
        prefix = sample_path.stem.rsplit("-", 1)[0] + "-"
        existing = sorted(sample_path.parent.glob(f"{prefix}*.tar"))
        if existing:
            existing_total = sum(p.stat().st_size for p in existing)
            print(f"  [tar_shard pack] {len(existing)} shards already exist "
                  f"({existing_total / 1e9:.2f} GB total) → skipping (use force=True to repack)")
            return {"n_files_packed": 0, "n_shards_written": len(existing),
                    "total_bytes": existing_total, "elapsed_sec": 0.0,
                    "max_shard_gb_observed": max(p.stat().st_size for p in existing) / 1e9,
                    "max_shard_size_gb": max_shard_size_gb}

    # Stream-pack: open shard, add files until next file would exceed cap, roll.
    t0 = time.time()
    n_files_packed = 0
    total_bytes = 0
    shard_idx = 0
    shard_path = _shard_path(shard_template, shard_idx)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    cur_tar = tarfile.open(shard_path, "w")
    cur_size = 0
    cur_count = 0
    written_paths: list = []

    def _close_current(tar, sp, size, count):
        tar.close()
        if count > 0:
            print(f"  shard {sp.stem.rsplit('-', 1)[-1]}: {count:5d} files, "
                  f"{size / 1e9:.3f} GB → {sp.name}")
            written_paths.append(sp)
        else:
            sp.unlink()  # remove empty shard

    for f in files:
        sz = f.stat().st_size
        # Roll over if adding this file would exceed cap (and current shard non-empty).
        if cur_count > 0 and cur_size + sz > max_bytes:
            _close_current(cur_tar, shard_path, cur_size, cur_count)
            shard_idx += 1
            shard_path = _shard_path(shard_template, shard_idx)
            cur_tar = tarfile.open(shard_path, "w")
            cur_size = 0
            cur_count = 0

        cur_tar.add(str(f), arcname=f.name)
        cur_size += sz
        cur_count += 1
        n_files_packed += 1
        total_bytes += sz

    # Close final shard.
    _close_current(cur_tar, shard_path, cur_size, cur_count)

    elapsed = time.time() - t0
    max_shard_gb_observed = max(
        (p.stat().st_size / 1e9 for p in written_paths), default=0.0)

    # Optional source cleanup
    if not keep_source:
        for f in files:
            f.unlink()
        print(f"  [tar_shard pack] deleted {len(files)} source files from {input_dir}/")
    else:
        print(f"  [tar_shard pack] --keep-source: left {len(files)} source files in "
              f"{input_dir}/ (downstream code reads individual files at runtime)")

    print(f"\n✅ tar_shard pack done: {n_files_packed} files → {len(written_paths)} shards, "
          f"{total_bytes / 1e9:.2f} GB in {elapsed:.0f}s "
          f"(max shard = {max_shard_gb_observed:.3f} GB, cap = {max_shard_size_gb:.2f} GB)")
    return {"n_files_packed": n_files_packed, "n_shards_written": len(written_paths),
            "total_bytes": total_bytes, "elapsed_sec": elapsed,
            "max_shard_gb_observed": max_shard_gb_observed,
            "max_shard_size_gb": max_shard_size_gb}


def unpack_shards_to_dir(
    shards_glob: str,
    output_dir: Path,
    skip_existing: bool,
) -> dict:
    """Extract all members from shards matching glob into output_dir/.

    Returns: {n_files_extracted, n_shards_read, total_bytes, elapsed_sec}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(shards_glob))
    if not shards:
        print(f"  [tar_shard unpack] no shards matched glob: {shards_glob}")
        return {"n_files_extracted": 0, "n_shards_read": 0, "total_bytes": 0, "elapsed_sec": 0.0}

    t0 = time.time()
    n_extracted = 0
    n_skipped = 0
    total_bytes = 0

    for shard_path in shards:
        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                target = output_dir / member.name
                if skip_existing and target.is_file():
                    n_skipped += 1
                    continue
                tar.extract(member, output_dir, filter="data")
                if target.is_file():
                    total_bytes += target.stat().st_size
                    n_extracted += 1
        print(f"  unpacked {Path(shard_path).name}: "
              f"{n_extracted} cum extracted, {n_skipped} cum skipped")

    elapsed = time.time() - t0
    print(f"\n✅ tar_shard unpack done: {n_extracted} files extracted "
          f"({n_skipped} skipped existing) from {len(shards)} shards, "
          f"{total_bytes / 1e9:.2f} GB in {elapsed:.0f}s → {output_dir}/")
    return {"n_files_extracted": n_extracted, "n_shards_read": len(shards),
            "total_bytes": total_bytes, "elapsed_sec": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TAR-shard utility for HF 10k-file limit (CPU-only).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── pack ──
    p_pack = sub.add_parser("pack", help="Pack input_dir/* → size-driven TAR shards")
    p_pack.add_argument("--input-dir", type=Path, required=True,
                        help="Directory of small files to pack (e.g., masks/)")
    p_pack.add_argument("--shard-template", required=True,
                        help='Output template, e.g., "masks-{shard:05d}.tar"')
    p_pack.add_argument("--max-shard-size-gb", type=float, default=None,
                        help="Soft size cap per shard in GB. Default: pipeline.yaml "
                             "data.max_tar_shard_gb (single source — iter18 H2 removed "
                             "the duplicated 1.0 literal). Files stream-fill the current "
                             "shard until adding the next would exceed this cap, "
                             "then a new shard rolls over (m00d-style auto-shard).")
    p_pack.add_argument("--keep-source", action="store_true",
                        help="Leave the per-file .npz/.npy on disk after packing "
                             "(default = delete to save disk; pass --keep-source if "
                             "downstream code reads individual files at runtime).")
    p_pack.add_argument("--force", action="store_true",
                        help="Repack even if shards exist (otherwise idempotent skip)")

    # ── unpack ──
    p_unpack = sub.add_parser("unpack", help="Extract tar shards → output_dir/")
    p_unpack.add_argument("--shards-glob", required=True,
                          help='Glob pattern matching shards, e.g., "masks-*.tar"')
    p_unpack.add_argument("--output-dir", type=Path, required=True,
                          help="Where to extract members (e.g., masks/ recreated)")
    # iter18 H2: dropped the dead `--skip-existing` flag (store_true with
    # default=True = a no-op switch). Skip-existing IS the documented default;
    # --force-overwrite is the one real toggle.
    p_unpack.add_argument("--force-overwrite", action="store_true",
                          help="Overwrite existing files (default: skip files that "
                               "already exist in output_dir)")

    args = parser.parse_args()

    if args.cmd == "pack":
        max_gb = args.max_shard_size_gb
        if max_gb is None:
            from utils.config import get_pipeline_config
            max_gb = get_pipeline_config()["data"]["max_tar_shard_gb"]
        pack_dir_to_shards(
            input_dir=args.input_dir,
            shard_template=args.shard_template,
            max_shard_size_gb=max_gb,
            keep_source=args.keep_source,
            force=args.force,
        )
    elif args.cmd == "unpack":
        skip = not args.force_overwrite
        unpack_shards_to_dir(
            shards_glob=args.shards_glob,
            output_dir=args.output_dir,
            skip_existing=skip,
        )


if __name__ == "__main__":
    # Avoid import-time failures from optional deps.
    main()
