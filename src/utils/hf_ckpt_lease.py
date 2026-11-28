"""Lease a checkpoint from HF for one eval run on a disk-capped box: `fetch` downloads it, `release` deletes it through
utils.cache_policy.guarded_delete (deleted only with --cache-policy 2). CPU-only.

Built for the video held-out test on the shared Hospital-Bench box (20 GB disk cap for factorjepa, Fri Nov 27 2026):
the four 1B arms need 8.3 GB of checkpoints each (ckpt_best + student_encoder), so they are fetched and released one
arm at a time. Re-downloadable copies of files on HF only; never point `release` at anything else.
USAGE:
    python -u src/utils/hf_ckpt_lease.py fetch --repo anonymousML123/factorjepa-outputs \\
        --path-in-repo outputs/poc/vjepa_2_1_vitg_1B/train/m09c_surgery_raw_encoder/m09c_ckpt_best.pt --dest-dir ckpt_lease
    python -u src/utils/hf_ckpt_lease.py release --dest-dir ckpt_lease \\
        --path-in-repo outputs/poc/vjepa_2_1_vitg_1B/train/m09c_surgery_raw_encoder/m09c_ckpt_best.pt --cache-policy 2
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.cache_policy import add_cache_policy_arg, guarded_delete  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--repo", required=True)
    f.add_argument("--path-in-repo", required=True)
    f.add_argument("--dest-dir", type=Path, required=True)
    r = sub.add_parser("release")
    r.add_argument("--path-in-repo", required=True)
    r.add_argument("--dest-dir", type=Path, required=True)
    add_cache_policy_arg(r)
    a = p.parse_args()
    if a.cmd == "fetch":
        out = hf_hub_download(a.repo, a.path_in_repo, repo_type="dataset", local_dir=a.dest_dir)
        size = Path(out).stat().st_size
        print(f"fetched {out} ({size / 1e9:.2f} GB)")
        return
    target = a.dest_dir / a.path_in_repo
    if not target.is_file():
        sys.exit(f"FATAL: {target} is not a leased file")
    if a.cache_policy is None:
        sys.exit("FATAL: release needs an explicit --cache-policy (2 deletes)")
    guarded_delete(target, str(a.cache_policy), label="leased checkpoint")


if __name__ == "__main__":
    main()
