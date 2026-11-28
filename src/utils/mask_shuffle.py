"""Random-view control (v4 Data2AgentEx judge ask "semantic views or masking"): give every training clip ANOTHER
clip's m10 mask.npz, so the layout / agent / interaction views keep realistic mask area, shape and tube statistics
but lose their meaning (a pedestrian mask lands on a road, an interaction tube crops an empty patch). CPU only.

Computed ONCE by run_train.sh (src/CLAUDE.md "shared derivation via CLI") and consumed by m09c1 via --mask-remap.
The remap is a seeded derangement WITHIN each factor-eligibility signature (has_D_L, has_D_A, has_D_I) from
factor_manifest.json, so the per-factor clip pools and their sizes are identical to the semantic arm and no clip
keeps its own mask. Domain = the same keys m09c1 indexes: --subset pool in manifest order, minus --val-subset,
with an m10 mask on disk. m09c1 asserts the domain matches its own mask index exactly (fail loud).

USAGE (run_train.sh builds these args from the train yaml mask_shuffle block + pipeline.yaml data paths):
    python -u src/utils/mask_shuffle.py \
        --factor-manifest data/eval_10k_local/m11_factor_datasets/factor_manifest.json \
        --masks-dir data/eval_10k_local/m10_sam_segment/masks \
        --subset data/eval_10k_local/train_pool.json --val-subset <val_split.json> \
        --seed <mask_shuffle.seed> --out outputs/poc/<bb>/train/<arm>/mask_remap.json
    python -u src/utils/mask_shuffle.py --selftest
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_subset  # noqa: E402

FACTORS = ("has_D_L", "has_D_A", "has_D_I")


def sattolo(keys: list, rng: np.random.Generator) -> dict:
    """Uniform random cyclic permutation (Sattolo 1986): a derangement for len >= 2, deterministic given rng."""
    n = len(keys)
    if n < 2:
        raise ValueError(f"mask_shuffle: a group of {n} clip cannot be deranged (it would keep its own mask)")
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))  # j < i: the Sattolo step (Fisher-Yates would allow j == i)
        perm[i], perm[j] = perm[j], perm[i]
    return {keys[i]: keys[perm[i]] for i in range(n)}


def build_remap(manifest: dict, keys: list, seed: int) -> dict:
    """Derangement of `keys` within each eligibility signature. Returns {clip_key: donor_clip_key}."""
    groups = defaultdict(list)
    for k in keys:
        groups[tuple(bool(manifest[k][f]) for f in FACTORS)].append(k)
    rng = np.random.default_rng(seed)
    remap = {}
    for sig in sorted(groups):
        remap.update(sattolo(groups[sig], rng))
    return remap, {"|".join(f"{f}={int(v)}" for f, v in zip(FACTORS, sig)): len(groups[sig]) for sig in sorted(groups)}


def domain_keys(manifest: dict, pool: set, val: set, masks_dir: Path) -> list:
    """The keys m09c1 indexes: pool ∩ manifest (manifest order) − val, with a mask on disk."""
    return [k for k in manifest if k in pool and k not in val
            and (masks_dir / f"{k.replace('/', '__')}.npz").exists()]


def apply_remap(mask_index: dict, remap_path: str) -> dict:
    """m09c1 side: point every clip at its donor's mask. Fails loud unless the remap covers exactly mask_index."""
    remap = json.loads(Path(remap_path).read_text())["remap"]
    a, b = set(mask_index), set(remap)
    if a != b:
        raise RuntimeError(f"mask_shuffle: remap domain != m09c1 mask index ({len(a - b)} clips missing from the remap, "
                           f"{len(b - a)} extra). Rebuild {remap_path} from the same --subset/--val-subset/masks.")
    fixed = [k for k, d in remap.items() if k == d]
    if fixed:
        raise RuntimeError(f"mask_shuffle: {len(fixed)} clips keep their own mask, e.g. {fixed[:2]}")
    return {k: mask_index[remap[k]] for k in mask_index}


def selftest() -> int:
    man = {f"c{i}": {"has_D_L": True, "has_D_A": i % 3 != 0, "has_D_I": i % 2 == 0} for i in range(40)}
    keys = list(man)
    r1, g = build_remap(man, keys, 7)
    r2, _ = build_remap(man, keys, 7)
    r3, _ = build_remap(man, keys, 8)
    checks = {
        "no fixed points": all(k != v for k, v in r1.items()),
        "bijection on the same key set": set(r1) == set(keys) and sorted(r1.values()) == sorted(keys),
        "deterministic for one seed": r1 == r2,
        "seed changes it": r1 != r3,
        "signature kept": all(tuple(man[k][f] for f in FACTORS) == tuple(man[v][f] for f in FACTORS) for k, v in r1.items()),
    }
    try:
        build_remap({"a": {"has_D_L": 1, "has_D_A": 0, "has_D_I": 0}}, ["a"], 0)
        checks["singleton group refused"] = False
    except ValueError:
        checks["singleton group refused"] = True
    idx = {k: Path(f"/m/{k}.npz") for k in keys}
    tmp = Path(__file__).resolve().parent.parent.parent / ".mask_shuffle_selftest.json"
    tmp.write_text(json.dumps({"remap": r1}))
    applied = apply_remap(idx, str(tmp))
    checks["apply points at donors"] = all(applied[k] == idx[r1[k]] for k in keys)
    try:
        apply_remap({**idx, "extra": Path("/m/extra.npz")}, str(tmp))
        checks["domain mismatch refused"] = False
    except RuntimeError:
        checks["domain mismatch refused"] = True
    tmp.unlink()
    for name, ok in checks.items():
        print(f"{'ok ' if ok else 'BAD'} {name}")
    print(f"groups {g}")
    print(f"selftest {sum(checks.values())} of {len(checks)}")
    return 0 if all(checks.values()) else 1


def main():
    if sys.argv[1:2] == ["--selftest"]:
        sys.exit(selftest())
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--factor-manifest", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--subset", required=True, help="leakage-safe train pool (clip_splits.py output)")
    ap.add_argument("--val-subset", required=True, help="the val split m09c1 subtracts (its --probe-subset)")
    ap.add_argument("--seed", type=int, required=True, help="train yaml mask_shuffle.seed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    manifest = json.loads(Path(a.factor_manifest).read_text())
    val = set(json.loads(Path(a.val_subset).read_text())["clip_keys"])
    keys = domain_keys(manifest, load_subset(a.subset), val, Path(a.masks_dir))
    if not keys:
        sys.exit(f"FATAL mask_shuffle: no clip of {a.subset} has a mask in {a.masks_dir}")
    remap, groups = build_remap(manifest, keys, a.seed)
    if any(k == d for k, d in remap.items()):
        sys.exit("FATAL mask_shuffle: a clip kept its own mask")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"seed": a.seed, "n": len(remap), "groups": groups,
                                       "factor_manifest": a.factor_manifest, "subset": a.subset,
                                       "val_subset": a.val_subset, "remap": remap}, indent=1))
    print(f"mask_shuffle: {len(remap)} clips deranged in {len(groups)} eligibility groups {groups} -> {a.out}")


if __name__ == "__main__":
    main()
