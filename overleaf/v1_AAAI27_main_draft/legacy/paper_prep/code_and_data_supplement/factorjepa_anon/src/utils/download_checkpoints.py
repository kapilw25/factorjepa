"""Download model checkpoints from the YAML registry. Idempotent, resumable (aria2c -c).

SINGLE SOURCE for both consumers of configs/checkpoints_download.yaml:
  · setup_env_uv.sh        → --group trainable      (the 2 backbones the scheduler trains; fast path
                             so the torch/GPU-deps install does NOT wait behind the eval-only ckpts)
  · §G frozen-baseline eval → --group eval_baselines (iter16 champion + image/version baselines; these
                             are loaded by run_eval.sh frozen_features/ijepa_features/lejepa_features →
                             feed the m12a/b/c/d/e/f aggregation that builds the §G frozen row)

The registry has a `catalog` (canonical filename → upstream URL) and `groups` (which filenames each
consumer fetches). A filename may belong to multiple groups; the URL is declared once in `catalog`.
The canonical filename is what run_eval.sh frozen_ckpt_for + configs/eval/probe_encoders.yaml expect;
aria2c -o renames differently-named upstream assets to it.

USAGE:
  # the 2 trainable backbones (setup_env_uv.sh fast path):
  python -u src/utils/download_checkpoints.py \
      --registry configs/checkpoints_download.yaml --group trainable --dest checkpoints

  # the §G frozen baselines (run on demand before re-evaluating the frozen row):
  python -u src/utils/download_checkpoints.py \
      --registry configs/checkpoints_download.yaml --group eval_baselines --dest checkpoints

  # everything (union of all groups):
  python -u src/utils/download_checkpoints.py \
      --registry configs/checkpoints_download.yaml --group all --dest checkpoints

  # preview only (resolve names→URLs, print plan, download nothing):
  python -u src/utils/download_checkpoints.py \
      --registry configs/checkpoints_download.yaml --group all --dest checkpoints --dry-run
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import yaml


def _resolve(registry_path: Path, group: str) -> dict:
    """Read the registry and return {canonical_filename: url} for the requested group.

    group == "all" → union of every group. FAIL LOUD on a missing registry / group / catalog entry.
    """
    reg = yaml.safe_load(registry_path.read_text())
    catalog = reg.get("catalog")
    groups = reg.get("groups")
    if not catalog or not groups:
        raise SystemExit(f"FATAL: {registry_path} must define both 'catalog' and 'groups' keys.")

    if group == "all":
        names = sorted({n for members in groups.values() for n in members})
    else:
        if group not in groups:
            raise SystemExit(f"FATAL: group '{group}' not in {registry_path} "
                             f"(have: {sorted(groups)} + 'all').")
        names = list(groups[group])

    resolved = {}
    for name in names:
        if name not in catalog:
            raise SystemExit(f"FATAL: '{name}' is listed in groups but missing from catalog in "
                             f"{registry_path}.")
        resolved[name] = catalog[name]
    return resolved


def _download_one(fname: str, url: str, dest: Path) -> str:
    """Fetch one checkpoint into dest/. Idempotent: a fully-downloaded file (present, no .aria2
    control sidecar) is skipped. Returns "skipped" | "downloaded". FAIL LOUD on a non-zero exit."""
    out = dest / fname
    sidecar = dest / f"{fname}.aria2"
    if out.exists() and out.stat().st_size > 0 and not sidecar.exists():
        print(f"  ✓ present: {fname} ({out.stat().st_size / 1e9:.1f} GB)", flush=True)
        return "skipped"

    print(f"  Downloading {fname} ← {url}", flush=True)
    if shutil.which("aria2c"):
        cmd = ["aria2c", "-c", "-x", "16", "-s", "16", "--auto-file-renaming=false",
               "-d", str(dest), "-o", fname, url]
    elif shutil.which("wget"):
        cmd = ["wget", "-c", "-O", str(out), url]
    else:
        raise SystemExit("FATAL: neither aria2c nor wget found on PATH for checkpoint download.")

    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"FATAL: download failed (rc={rc}) for {fname} from {url}")
    return "downloaded"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", required=True,
                    help="YAML registry with 'catalog' (filename→url) + 'groups' (membership)")
    ap.add_argument("--group", required=True, choices=["trainable", "eval_baselines", "all"],
                    help="which group to fetch (or 'all' for the union)")
    ap.add_argument("--dest", required=True, help="destination directory (e.g. checkpoints)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve + print the plan, download nothing")
    args = ap.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.is_file():
        raise SystemExit(f"FATAL: registry not found: {registry_path}")

    items = _resolve(registry_path, args.group)
    dest = Path(args.dest)
    print(f"Checkpoint download [group={args.group}] → {dest}/ : {len(items)} file(s)", flush=True)
    for fname, url in items.items():
        print(f"  - {fname}  ←  {url}")

    if args.dry_run:
        print("[dry-run] resolved OK — downloaded nothing.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    n_skipped = n_downloaded = 0
    for i, (fname, url) in enumerate(items.items(), 1):
        print(f"[{i}/{len(items)}]", end=" ", flush=True)
        if _download_one(fname, url, dest) == "skipped":
            n_skipped += 1
        else:
            n_downloaded += 1

    present = len(list(dest.glob("*.pt"))) + len(list(dest.glob("*.pth.tar")))
    print(f"Done [group={args.group}]: {n_downloaded} downloaded, {n_skipped} already present "
          f"({present} total ckpt files in {dest}/).", flush=True)


if __name__ == "__main__":
    main()
