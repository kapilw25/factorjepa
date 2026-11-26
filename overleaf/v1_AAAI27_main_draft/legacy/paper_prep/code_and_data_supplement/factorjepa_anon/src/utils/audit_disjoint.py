"""Disjointness audit between two clip manifests (iter18 cross-set CI-reduction).

The tighter confidence bands of the cross-set eval (probe heads trained on `eval_10k`, tested on
the FULL disjoint `subset_10k`) are only CLEAN if the two corpora share no clip and no same-scene
near-duplicate — both were carved from the same DenseWorld parent, so the silent failure mode is a
re-sampled clip leaking the train corpus into the test corpus. This module proves disjointness,
FAIL LOUD, at three strengths:

  exact_overlap      : identical clip KEY in both sets                      → hard leakage
  shared_source_video: same SOURCE VIDEO id in both sets (any two clips of  → same-scene leakage
                       the same street/camera) — the strongest clean claim
  adjacent_clips     : same source video AND clip indices within ±window    → the ±30 s hard-mode
                       (consecutive slices of one video = near-identical)     rule, metadata-based

It is METADATA-based (clip key = `<tier>/<city>/<cat>/<video_id>/<video_id>-<NNN>.mp4`), NOT
frame-embedding cosine: consecutive walking clips from the same camera are naturally > 0.90 cosine,
so an embedding pass floods false "duplicates" (iter18 plan fix-4). Pure-CPU, stdlib only.
**No `m*.py` import (rule 32)**; the ±window is a REQUIRED CLI arg (no hardcoded config default).

USAGE (real manifests — eval_10k carries `clip_keys`, subset_10k's manifest carries `saved_keys`):
    source venv_denseworld/bin/activate
    python -u src/utils/audit_disjoint.py \
        --set-a data/eval_10k_local/eval_10k.json        --keys-field-a clip_keys  --label-a eval_10k \
        --set-b data/subset_10k_local/manifest.json      --keys-field-b saved_keys --label-b subset_10k \
        --window-clips 6

USAGE (synthetic self-contained smoke):
    python -u src/utils/audit_disjoint.py --selftest --window-clips 6
"""
import argparse
import json
import sys
from pathlib import Path

_SAMPLE_N = 10   # offending ids/pairs printed before "... and N more"


def parse_clip_key(key):
    """`<tier>/<city>/<cat>/<video_id>/<video_id>-<NNN>.mp4` → (video_id, clip_index:int).

    video_id = the PARENT directory name (robust: groups all clips of one source video).
    clip_index = the trailing `-NNN` ordinal of the filename. Returns (key, None) for a key that
    doesn't end in `-<digits>.mp4` so the caller can still exact-match it but skips it in the
    index-adjacency layer (FAIL LOUD is the loader's job, not this pure parser's)."""
    stem = key[:-4] if key.endswith(".mp4") else key
    parts = stem.split("/")
    base = parts[-1]                              # "<video_id>-NNN"
    video_id = parts[-2] if len(parts) >= 2 else base
    if "-" in base and base.rsplit("-", 1)[1].isdigit():
        return video_id, int(base.rsplit("-", 1)[1])
    return video_id, None


def exact_overlap(keys_a, keys_b):
    """Identical clip-KEY in both corpora. Returns {n_a, n_b, n_overlap, sample}."""
    sa, sb = set(keys_a), set(keys_b)
    shared = sorted(sa & sb)
    return {"n_a": len(sa), "n_b": len(sb), "n_overlap": len(shared), "sample": shared[:_SAMPLE_N]}


def shared_source_videos(keys_a, keys_b):
    """Source-video ids appearing in BOTH corpora. Returns {n_videos_a, n_videos_b, n_shared, sample}."""
    va = {parse_clip_key(k)[0] for k in keys_a}
    vb = {parse_clip_key(k)[0] for k in keys_b}
    shared = sorted(va & vb)
    return {"n_videos_a": len(va), "n_videos_b": len(vb),
            "n_shared": len(shared), "sample": shared[:_SAMPLE_N]}


def adjacent_clips(keys_a, keys_b, window_clips):
    """Cross-corpus pairs in the SAME source video whose clip indices are within ±window_clips.

    window_clips : int (REQUIRED — a config value, never defaulted in-signature). Per shared video the
    indices are sorted and swept with a two-pointer window → ~O(n log n), safe at 10k × 10k. Returns a
    sorted list of {video, idx_a, idx_b, gap}."""
    def by_vid(keys):
        d = {}
        for k in keys:
            v, i = parse_clip_key(k)
            if i is not None:
                d.setdefault(v, []).append(i)
        return d
    a_by, b_by = by_vid(keys_a), by_vid(keys_b)
    out = []
    for v in set(a_by) & set(b_by):
        a_sorted, b_sorted = sorted(a_by[v]), sorted(b_by[v])
        lo = 0
        for ia in a_sorted:
            while lo < len(b_sorted) and b_sorted[lo] < ia - window_clips:
                lo += 1
            j = lo
            while j < len(b_sorted) and b_sorted[j] <= ia + window_clips:
                out.append({"video": v, "idx_a": ia, "idx_b": b_sorted[j],
                            "gap": abs(ia - b_sorted[j])})
                j += 1
    out.sort(key=lambda p: (p["gap"], p["video"], p["idx_a"], p["idx_b"]))
    return out


def _load_keys(path, keys_field):
    """Load clip keys from a manifest json. keys_field names the list of key-strings
    (e.g. 'clip_keys' for eval_10k.json, 'saved_keys' for a *_local/manifest.json).
    FAIL LOUD if the field is absent or not a list of strings — a vacuous audit is worse than a crash."""
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, dict) or keys_field not in obj:
        sys.exit(f"FATAL: {path} has no top-level '{keys_field}' list "
                 f"(top-keys: {sorted(obj)[:10] if isinstance(obj, dict) else type(obj).__name__}) "
                 f"— wrong --keys-field?")
    keys = obj[keys_field]
    if not isinstance(keys, list) or not keys or not isinstance(keys[0], str):
        sys.exit(f"FATAL: {path}['{keys_field}'] is not a non-empty list of strings")
    return keys


def run_audit(keys_a, keys_b, window_clips, label_a, label_b):
    """Run all three checks + print the report. Returns True iff CLEAN (0 exact, 0 adjacent)."""
    ex = exact_overlap(keys_a, keys_b)
    sv = shared_source_videos(keys_a, keys_b)
    adj = adjacent_clips(keys_a, keys_b, window_clips)

    print(f"disjointness audit: {label_a} ({ex['n_a']} clips / {sv['n_videos_a']} videos)  vs  "
          f"{label_b} ({ex['n_b']} clips / {sv['n_videos_b']} videos)")
    print(f"  [exact]    shared clip keys              : {ex['n_overlap']}")
    for s in ex["sample"]:
        print(f"               − {s}")
    print(f"  [video]    shared source videos          : {sv['n_shared']}")
    for s in sv["sample"]:
        print(f"               − {s}")
    print(f"  [adjacent] same video within ±{window_clips} clips    : {len(adj)} cross pair(s)")
    for p in adj[:_SAMPLE_N]:
        print(f"               − {p['video']}: clip {p['idx_a']} ~ {p['idx_b']}  (gap={p['gap']})")
    if len(adj) > _SAMPLE_N:
        print(f"               ... and {len(adj) - _SAMPLE_N} more")

    clean = (ex["n_overlap"] == 0 and len(adj) == 0)
    if not clean:
        verdict = "DIRTY — leakage between the corpora; the tighter cross-set CIs would be contaminated"
    elif sv["n_shared"] == 0:
        verdict = (f"CLEAN — 0 shared clips · 0 shared source videos "
                   f"(strongest disjointness; the cross-set CIs are leakage-free)")
    else:
        verdict = (f"CLEAN — 0 shared clips · 0 adjacent (±{window_clips}) clips, but {sv['n_shared']} "
                   f"source video(s) appear in both with non-adjacent clips (note in the appendix)")
    print(f"  → {verdict}")
    return clean


def _selftest(window_clips):
    """Synthetic smoke over the real path-key format: a clean pair passes; an injected shared key
    and an injected adjacent same-video clip are both caught."""
    a = ["goa/walking/VIDx/VIDx-004.mp4", "goa/walking/VIDx/VIDx-010.mp4", "tier2/bhopal/rain/VIDy/VIDy-002.mp4"]
    b = ["goa/walking/VIDz/VIDz-004.mp4", "tier1/blr/drone/VIDw/VIDw-100.mp4"]
    assert exact_overlap(a, b)["n_overlap"] == 0, "clean keys flagged"
    assert shared_source_videos(a, b)["n_shared"] == 0, "clean videos flagged"
    assert adjacent_clips(a, b, window_clips) == [], "clean corpora flagged adjacent"
    assert parse_clip_key("goa/walking/VIDx/VIDx-004.mp4") == ("VIDx", 4), "key parse wrong"
    print(f"[selftest] clean corpora → PASS (0 exact, 0 shared-video, 0 adjacent within ±{window_clips})")

    b_dirty = b + ["goa/walking/VIDx/VIDx-004.mp4",      # exact-duplicate key
                   "goa/walking/VIDx/VIDx-012.mp4"]      # same video VIDx, 2 clips from idx 10 → adjacent
    assert exact_overlap(a, b_dirty)["n_overlap"] == 1, "missed shared key"
    sv = shared_source_videos(a, b_dirty)
    adj = adjacent_clips(a, b_dirty, window_clips)
    assert sv["n_shared"] == 1 and sv["sample"] == ["VIDx"], f"missed shared video: {sv}"
    assert any(p["video"] == "VIDx" and p["gap"] <= window_clips for p in adj), f"missed adjacent: {adj}"
    print(f"[selftest] injected leakage → caught 1 shared key + {sv['n_shared']} shared video + "
          f"{len(adj)} adjacent pair(s)")
    print("[selftest] PASS — exact + shared-video + adjacency checks all fire correctly")


def main():
    ap = argparse.ArgumentParser(description="cross-set clip-disjointness audit (iter18 CI-reduction)")
    ap.add_argument("--selftest", action="store_true", help="synthetic smoke (no data needed)")
    ap.add_argument("--set-a", help="manifest json A (the train corpus, e.g. eval_10k.json)")
    ap.add_argument("--set-b", help="manifest json B (the test corpus, e.g. subset_10k manifest.json)")
    ap.add_argument("--keys-field-a", help="name of the clip-key list in A (e.g. clip_keys)")
    ap.add_argument("--keys-field-b", help="name of the clip-key list in B (e.g. saved_keys)")
    ap.add_argument("--label-a", default="set_A", help="display label for A")
    ap.add_argument("--label-b", default="set_B", help="display label for B")
    ap.add_argument("--window-clips", type=int, required=True,
                    help="adjacency window in clip-index units (consecutive slices of one source video)")
    args = ap.parse_args()

    if args.selftest:
        _selftest(args.window_clips)
        return
    for req, val in [("--set-a", args.set_a), ("--set-b", args.set_b),
                     ("--keys-field-a", args.keys_field_a), ("--keys-field-b", args.keys_field_b)]:
        if not val:
            ap.error(f"{req} is required for a real audit (or pass --selftest)")
    ka = _load_keys(args.set_a, args.keys_field_a)
    kb = _load_keys(args.set_b, args.keys_field_b)
    clean = run_audit(ka, kb, args.window_clips, args.label_a, args.label_b)
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
