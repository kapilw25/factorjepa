#!/usr/bin/env python3
"""demo_frames — manage the throwaway per-frame PNG scaffolding that the demo render scripts
(m14/m16/m17 and the scratch renderers) write as `frames_<timestamp>/` dirs — one PNG per video
frame — before ffmpeg stitches the `.mp4`.

The PNGs are pure INTERMEDIATES; the `.mp4` is the deliverable, so they never need backing up.
They are regenerable two ways:
  1. re-run the render script  — recomputes from the model (the authoritative rebuild), OR
  2. `regen` here              — ffmpeg-extract them straight from the finished `.mp4` (cheap, no GPU).

  clean : remove every `frames_*` dir that has a SIBLING `.mp4` (safe-by-construction — if the mp4
          isn't there the render may have failed, so we keep the frames). Dry-run by default;
          `--apply` actually deletes. This is what shrinks HF/git backups (drops the ~10.7k-PNG swarm
          that trips HF's LFS rate limiter).
  regen : rebuild the PNGs from an mp4 into `<stem>_frames/` (does NOT match `frames_*`, so a later
          `clean` won't re-delete it).

No shell `rm` (CLAUDE.md delete-protection): deletes live here in .py, gated by `--apply` + the
sibling-mp4 safety check (mirrors utils/clear_resume_anchors deleting scratch only when it's safe).

USAGE:
    python src/utils/demo_frames.py clean --root outputs/demo                  # dry-run: list what would go
    python src/utils/demo_frames.py clean --root outputs/demo --apply          # delete (keeps every .mp4)
    python src/utils/demo_frames.py regen --mp4 outputs/demo/mcq/demo_mcq.mp4  # rebuild frames from the mp4
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def _find_frames_dirs(root):
    return sorted(p for p in root.rglob("frames_*") if p.is_dir())


def _dir_stats(p):
    files = [f for f in p.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def clean(root, apply):
    if not root.exists():
        sys.exit(f"FATAL: root not found: {root}")
    dirs = _find_frames_dirs(root)
    if not dirs:
        print(f"[demo_frames] no frames_* dirs under {root} — nothing to clean")
        return
    removable, total_n, total_sz = [], 0, 0
    for d in dirs:
        mp4s = [m for m in d.parent.glob("*.mp4") if m.stat().st_size > 0]   # real encoded video, not a 0-byte temp
        n, sz = _dir_stats(d)
        if mp4s:
            removable.append((d, n, sz, mp4s[0].name)); total_n += n; total_sz += sz
        else:
            print(f"  SKIP  {d}  ({n} PNGs) — NO sibling .mp4 (render may have failed → kept)")
    for d, n, sz, mp4 in removable:
        print(f"  {'DELETE' if apply else 'would delete'}  {d}  ({n} PNGs, {sz/1e6:.0f} MB) — mp4 present: {mp4}")
    tail = "DELETED" if apply else "to delete (dry-run — pass --apply)"
    print(f"[demo_frames] {len(removable)} dir(s) · {total_n} PNGs · {total_sz/1e9:.2f} GB {tail}")
    if apply:
        for d, *_ in tqdm(removable, desc="removing frames_*"):
            shutil.rmtree(d)
        print(f"[demo_frames] freed {total_sz/1e9:.2f} GB · re-run the render (or `regen`) to rebuild any frames")


def regen(mp4, out):
    if not mp4.exists():
        sys.exit(f"FATAL: mp4 not found: {mp4}")
    out = out or (mp4.parent / f"{mp4.stem}_frames")          # NOT frames_* → safe from a later `clean`
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), str(out / "%06d.png")], check=True)
    n = len(list(out.glob("*.png")))
    if n == 0:
        sys.exit(f"FATAL: ffmpeg produced 0 frames from {mp4}")
    print(f"[demo_frames] regenerated {n} PNGs from {mp4.name} → {out}")


def main():
    p = argparse.ArgumentParser(description="manage throwaway demo frame PNGs (clean / regen)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clean", help="delete frames_* dirs that already have their .mp4")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--apply", action="store_true", help="actually delete (default = dry-run list)")
    r = sub.add_parser("regen", help="rebuild frames from a finished .mp4")
    r.add_argument("--mp4", type=Path, required=True)
    r.add_argument("--out", type=Path, default=None, help="frames dir (default <stem>_frames/ beside the mp4)")
    a = p.parse_args()
    if a.cmd == "clean":
        clean(a.root, a.apply)
    else:
        regen(a.mp4, a.out)


if __name__ == "__main__":
    main()
