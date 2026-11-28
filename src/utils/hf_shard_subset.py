"""Cut a clip subset out of the corpus TAR shards on HF by streaming, without storing the source shards. CPU-only.

The corpus shards (e.g. data/full_local/m00d_download_subset/subset-*.tar in anonymousML123/factorjepa-outputs) hold
`<index>.mp4` + `<index>.json` pairs, where <index> is the clip's position in the corpus manifest (full_local.json
clip_keys) and each shard holds one contiguous index range (about 1000 clips; sizes vary). This reads the first member
of every shard (one 512-byte range request each) to map ranges to shards, then streams only the shards that hold wanted clips,
stops each stream after its last wanted clip, checks every kept clip's sidecar against the corpus key, and writes
<output-dir>/m00d_download_subset/subset-NNNNN.tar plus <output-dir>/manifest.json in the layout the m12* readers use.
Fails loud if any wanted clip is not found. Public repo, no token needed.

Built for the video held-out test (FactorJEPA workshop drafts, Fri Nov 27 2026) on a box with a 20 GB disk cap.
USAGE:
    python -u src/utils/hf_shard_subset.py \\
        --repo anonymousML123/factorjepa-outputs --shard-prefix data/full_local/m00d_download_subset/ \\
        --corpus data/full_local/full_local.json --subset data/heldout_video/heldout_video.json \\
        --output-dir data/heldout_video_local --workers 8 \\
        2>&1 | tee logs/hf_shard_subset_$(date +%Y%m%d_%H%M%S).log
"""
import argparse
import bisect
import io
import json
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_paths import shards_dir  # noqa: E402
from utils.data_paths import artifact  # noqa: E402

def first_index(url: str) -> int:
    """Corpus index of a shard's first member (one 512-byte range request)."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-511"})
    head = urllib.request.urlopen(req, timeout=120).read()
    name = head[:100].split(b"\0")[0].decode()
    return int(name.split(".")[0])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True)
    p.add_argument("--shard-prefix", required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, required=True, help="parallel HF streams (network-bound)")
    a = p.parse_args()

    keys = json.loads(a.corpus.read_text())["clip_keys"]
    pos = {k: i for i, k in enumerate(keys)}
    want_keys = json.loads(a.subset.read_text())["clip_keys"]
    missing = [k for k in want_keys if k not in pos]
    if missing:
        sys.exit(f"FATAL: {len(missing)} subset keys are not in the corpus, e.g. {missing[:2]}")
    want = {pos[k] for k in want_keys}

    files = sorted(f for f in HfApi().list_repo_files(a.repo, repo_type="dataset")
                   if f.startswith(a.shard_prefix) and f.endswith(".tar"))
    urls = [hf_hub_url(a.repo, f, repo_type="dataset") for f in files]
    with ThreadPoolExecutor(a.workers) as ex:  # network-bound range requests, not tensor work
        starts = list(ex.map(first_index, urls))
    # Each shard holds a contiguous index range starting at its first member; a shard ends where the next one
    # starts (shard sizes vary: subset-00113 starts at 113313). A wrong layout surfaces as a FATAL missing clip.
    order = sorted(range(len(starts)), key=lambda j: starts[j])
    bounds = [starts[j] for j in order]
    if len(set(bounds)) != len(bounds):
        sys.exit("FATAL: two shards start at the same corpus index")
    by_shard = {}
    for i in want:
        j = order[bisect.bisect_right(bounds, i) - 1]
        if i < starts[j]:
            sys.exit(f"FATAL: corpus index {i} lies before the first shard")
        by_shard.setdefault(j, set()).add(i)
    blocks = sorted(by_shard)
    print(f"{len(want)} wanted clips in {len(blocks)} of {len(files)} shards; {a.workers} parallel streams", flush=True)

    out_dir = shards_dir(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def cut_block(b: int) -> tuple:
        """Stream source shard b; write its wanted clips to subset-<shard>.tar. Returns (shard name, counts)."""
        need = by_shard[b]
        got, kept, name = set(), {}, f"subset-{b:05d}.tar"
        with urllib.request.urlopen(urls[b], timeout=600) as resp, \
                tarfile.open(fileobj=resp, mode="r|") as src, tarfile.open(out_dir / name, "w") as out:
            for m in src:
                idx = int(m.name.split(".")[0])
                if idx not in need:
                    continue
                data = src.extractfile(m).read()
                if m.name.endswith(".json"):
                    side = json.loads(data)
                    key = f"{side['section']}/{side['video_id']}/{side['source_file']}"
                    if key != keys[idx]:
                        raise RuntimeError(f"sidecar {m.name} says {key}, corpus index {idx} is {keys[idx]}")
                    got.add(idx)
                info = tarfile.TarInfo(m.name)
                info.size = len(data)
                out.addfile(info, io.BytesIO(data))
                kept[idx] = kept.get(idx, 0) + 1
                if got >= need and all(kept.get(i, 0) == 2 for i in need):
                    break
        print(f"block {b}: kept {len(got)}/{len(need)} clips", flush=True)
        return name, kept

    shards, kept = [], {}
    with ThreadPoolExecutor(a.workers) as ex:  # network-bound streams
        for name, k in ex.map(cut_block, blocks):
            shards.append(name)
            kept.update(k)
    short = [i for i in want if kept.get(i, 0) != 2]
    if short:
        sys.exit(f"FATAL: {len(short)} wanted clips missing an mp4 or json, e.g. {[keys[i] for i in short[:3]]}")
    manifest = {"shards": sorted(shards), "n_clips": len(want), "subset": str(a.subset), "method": "hf_shard_subset",
                "source_repo": a.repo, "source_prefix": a.shard_prefix}
    (a.output_dir / artifact("manifest")).write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(want)} clips in {len(shards)} shards to {out_dir}")


if __name__ == "__main__":
    main()
