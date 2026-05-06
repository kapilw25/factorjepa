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
    n_shards: int,
    keep_source: bool,
    force: bool,
) -> dict:
    """Pack all files in input_dir into N TAR shards.

    Returns: {n_files_packed, n_shards_written, total_bytes, elapsed_sec}.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        print(f"FATAL: input_dir not a directory: {input_dir}")
        sys.exit(1)

    # Collect candidate files (top-level only; nested dirs not supported here —
    # caller should pass leaf .npz/.npy dirs like masks/ or D_L/, not parent).
    files = sorted(p for p in input_dir.iterdir() if p.is_file())
    if not files:
        print(f"  [tar_shard pack] no files under {input_dir} — skipping")
        return {"n_files_packed": 0, "n_shards_written": 0, "total_bytes": 0, "elapsed_sec": 0.0}

    # Idempotency: if all N shards exist + non-zero, skip
    existing_shards = [_shard_path(shard_template, i) for i in range(n_shards)]
    if not force and all(p.is_file() and p.stat().st_size > 0 for p in existing_shards):
        existing_total = sum(p.stat().st_size for p in existing_shards)
        print(f"  [tar_shard pack] all {n_shards} shards already exist "
              f"({existing_total / 1e9:.2f} GB total) → skipping (use --force to repack)")
        return {"n_files_packed": 0, "n_shards_written": 0,
                "total_bytes": existing_total, "elapsed_sec": 0.0}

    # Bucket files into shards via stable hash
    buckets: list = [[] for _ in range(n_shards)]
    for f in files:
        idx = _stable_shard_index(f.name, n_shards)
        buckets[idx].append(f)

    # Open all shards for write — overwrite if --force, append-skip-existing otherwise
    t0 = time.time()
    n_files_packed = 0
    total_bytes = 0
    n_shards_written = 0

    for shard_idx, bucket in enumerate(buckets):
        shard_path = _shard_path(shard_template, shard_idx)
        if not bucket:
            print(f"  shard {shard_idx:05d}: empty → skipping write")
            continue

        # Determine existing members (skip on incremental resume)
        existing_members: set = set()
        if shard_path.is_file() and not force:
            try:
                with tarfile.open(shard_path, "r") as tar_ro:
                    existing_members = {m.name for m in tar_ro.getmembers()}
            except tarfile.TarError:
                # Corrupt → start fresh
                existing_members = set()

        # Open in append mode if shard exists, else write
        mode = "a" if (shard_path.is_file() and not force and existing_members) else "w"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(shard_path, mode) as tar:
            for f in bucket:
                if f.name in existing_members:
                    continue
                tar.add(str(f), arcname=f.name)
                n_files_packed += 1
                total_bytes += f.stat().st_size

        n_shards_written += 1
        sz = shard_path.stat().st_size
        print(f"  shard {shard_idx:05d}: {len(bucket):4d} files, "
              f"{sz / 1e9:.2f} GB → {shard_path}")

    elapsed = time.time() - t0

    # Optional source cleanup (--keep-source flips this off)
    if not keep_source:
        for f in files:
            f.unlink()
        print(f"  [tar_shard pack] deleted {len(files)} source files from {input_dir}/")
    else:
        print(f"  [tar_shard pack] --keep-source: left {len(files)} source files in "
              f"{input_dir}/ (m11 + m09c read .npz directly during training)")

    print(f"\n✅ tar_shard pack done: {n_files_packed} files → {n_shards_written} shards, "
          f"{total_bytes / 1e9:.2f} GB in {elapsed:.0f}s")
    return {"n_files_packed": n_files_packed, "n_shards_written": n_shards_written,
            "total_bytes": total_bytes, "elapsed_sec": elapsed}


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
    p_pack = sub.add_parser("pack", help="Pack input_dir/* → N tar shards")
    p_pack.add_argument("--input-dir", type=Path, required=True,
                        help="Directory of small files to pack (e.g., masks/)")
    p_pack.add_argument("--shard-template", required=True,
                        help='Output template, e.g., "masks-{shard:05d}.tar"')
    p_pack.add_argument("--n-shards", type=int, required=True,
                        help="Number of shards (HF allows ~10k files; pick so files-per-shard ≤ 1500)")
    p_pack.add_argument("--keep-source", action="store_true",
                        help="Leave the per-file .npz/.npy on disk after packing "
                             "(default = delete to save disk; pass --keep-source if "
                             "downstream code reads individual files at runtime).")
    p_pack.add_argument("--force", action="store_true",
                        help="Repack even if all shards exist (otherwise idempotent skip)")

    # ── unpack ──
    p_unpack = sub.add_parser("unpack", help="Extract tar shards → output_dir/")
    p_unpack.add_argument("--shards-glob", required=True,
                          help='Glob pattern matching shards, e.g., "masks-*.tar"')
    p_unpack.add_argument("--output-dir", type=Path, required=True,
                          help="Where to extract members (e.g., masks/ recreated)")
    p_unpack.add_argument("--skip-existing", action="store_true", default=True,
                          help="Skip files that already exist in output_dir (default ON)")
    p_unpack.add_argument("--force-overwrite", action="store_true",
                          help="Overwrite existing files (sets skip_existing=False)")

    args = parser.parse_args()

    if args.cmd == "pack":
        pack_dir_to_shards(
            input_dir=args.input_dir,
            shard_template=args.shard_template,
            n_shards=args.n_shards,
            keep_source=args.keep_source,
            force=args.force,
        )
    elif args.cmd == "unpack":
        skip = args.skip_existing and not args.force_overwrite
        unpack_shards_to_dir(
            shards_glob=args.shards_glob,
            output_dir=args.output_dir,
            skip_existing=skip,
        )


if __name__ == "__main__":
    # Avoid import-time failures from optional deps.
    main()
