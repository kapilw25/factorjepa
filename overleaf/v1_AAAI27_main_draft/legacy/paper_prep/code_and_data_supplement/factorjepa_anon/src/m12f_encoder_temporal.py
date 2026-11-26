"""m12f — Encoder-temporal metric suite (EVALUATION band). GPU-only.

Four encoder-feature-based temporal metrics (plan_metrics_temporal.md §6) — none read the
predictor; all read frozen-or-surgical ENCODER features. Trainable heads for AoT/TOV/Pace
(tiny linear, train on train-split features, eval per-clip on test); TCC is training-free
(soft-NN cycle-back alignment on per-frame features).

  --metric aot   Arrow-of-Time  (Wei CVPR18)            ★ direction
  --metric tov   TOV / VCOP     (Misra ECCV16 / Xu CVPR19) ★ order
  --metric pace  Pace           (Wang ECCV20)             ★ rate
  --metric tcc   TCC (training-free) (Dwibedi CVPR19)    ★ correspondence

ONE standalone entry: loads the encoder ONCE per variant (rule-32 + load cost) via
utils.predictor_eval.load_encoder_only (no predictor build — iter16 §3.3 R2).
Dispatches --metric {one|all} to the et_*.py util fns. Mirrors m12e_predictor_temporal.py's
forward + paired_per_variant structure. Pace requires OVERSAMPLE decode (--pace-source-frames
= num_frames × max(strides)); the orchestrator switches the decode frame-count for the pace
pass only.

ALL sweep hyperparameters (n_permutations, strides, head LR/epochs/WD/BS, TCC temperature)
arrive via argparse — NO module-level constants per CLAUDE.md "no hardcoded values in Python".
The thin shell (run_eval.sh) will source them from `configs/pipeline.yaml probe.encoder_temporal`
at §3.3 integration.

Gold standard: docstring of each utils/et_*.py — Wei CVPR18, Misra ECCV16, Xu CVPR19, Wang
ECCV20, Dwibedi CVPR19.
GPU-VALIDATION REQUIRED post-eval (§3.1) — written CPU-checked; eval holds GPU now.

USAGE:
    python -u src/m12f_encoder_temporal.py --POC --stage forward --metric all \\
        --variant vjepa_2_1_frozen --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \\
        --action-probe-root outputs/poc/probe_action --local-data data/eval_10k_local \\
        --output-root outputs/poc/encoder_temporal \\
        --tubelet-size 2 --tov-n-permutations 4 --pace-strides 1,2,4 \\
        --pace-source-frames 64 --tcc-temperature 0.1 --tcc-pair-chunk auto \\
        --share-features 1 \\
        --head-lr 1e-3 --head-epochs 20 --head-weight-decay 1e-4 --head-batch-size 64 \\
        --cache-policy 1 --no-wandb
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import et_aot, et_pace, et_tcc, et_tov  # noqa: E402

from utils.action_labels import load_action_labels  # noqa: E402
from utils.bootstrap import parallel_bca  # noqa: E402
from utils.cache_policy import (  # noqa: E402
    add_cache_policy_arg, guarded_delete, resolve_cache_policy_interactive,
)
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog  # noqa: E402
from utils.checkpoint import save_array_checkpoint, save_json_checkpoint  # noqa: E402
from utils.config import add_local_data_arg, check_gpu, get_pipeline_config  # noqa: E402
from utils.data_download import ensure_local_data, iter_clips_parallel  # noqa: E402
from utils.decode_feeder import BatchFeeder, DecodeFeeder  # noqa: E402
from utils.gpu_gather import device_gather  # noqa: E402
from utils.frozen_features import ENCODERS, decode_to_tensor  # noqa: E402
from utils.gpu_batch import cleanup_temp, cuda_cleanup  # noqa: E402
from utils.predictor_eval import CROP, NUM_FRAMES_DEFAULT, bootstrap_ci, load_encoder_only  # noqa: E402
from utils.progress import make_pbar  # noqa: E402
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics  # noqa: E402

_PCFG = get_pipeline_config()
CHECKPOINT_EVERY = _PCFG["streaming"]["checkpoint_every"]

# metric → (lower_is_better, interpretation). TCC writes 2 per-clip arrays.
METRICS = {
    "aot":  (False, "Arrow-of-Time binary acc; higher = encoder preserves temporal direction"),
    "tov":  (False, "TOV/VCOP top-1 over n_permutations; higher = encoder retains frame order"),
    "pace": (False, "Pace top-1 over playback rates; higher = encoder is rate-sensitive"),
    "tcc":  (None,  "TCC cycle-back error (lower better) + Kendall's τ (higher better)"),
}


def _resolve_metrics(metric_arg):
    """Expand --metric all → the 4 metrics, then DROP any listed in ET_SKIP_METRICS (comma list). Lets
    the orchestrator skip an expensive metric without a code change — e.g. ET_SKIP_METRICS=pace, since
    pace needs an uncacheable nf64 oversample-decode (~28 s/clip cold) that dwarfs aot/tov/tcc. A single
    --metric is returned verbatim (the skip-list is ignored — you asked for exactly that one)."""
    if metric_arg != "all":
        return [metric_arg]                              # explicit single metric → run exactly it (skip-list ignored)
    skip = {m.strip() for m in os.environ.get("ET_SKIP_METRICS", "").split(",") if m.strip()}
    return [m for m in METRICS if m not in skip]

# V-JEPA-only registry mirror (image-JEPAs skipped — they have no native temporal axis here).
KNOWN_VARIANTS = tuple(n for n, s in ENCODERS.items() if s.get("kind") == "vjepa")


def _gen_permutations(n_permutations, T, seed):
    """Identity + (n_permutations - 1) deterministic non-identity permutations of length T.
    FAIL LOUD if T is too small to provide enough distinct non-identity permutations."""
    if n_permutations < 2:
        raise SystemExit(f"FATAL: --tov-n-permutations must be >= 2; got {n_permutations}")
    if T < 3:
        raise SystemExit(f"FATAL: T={T} too small for TOV (need >=3 frames)")
    perms = [torch.arange(T)]
    seen = {tuple(perms[0].tolist())}
    g = torch.Generator(device="cpu").manual_seed(seed)
    attempts = 0
    while len(perms) < n_permutations:
        cand = torch.randperm(T, generator=g)
        key = tuple(cand.tolist())
        if key not in seen:
            perms.append(cand)
            seen.add(key)
        attempts += 1
        if attempts > 10_000:
            raise SystemExit(
                f"FATAL: could not draw {n_permutations} distinct permutations for T={T}")
    return perms


def _run_metric_forward(metric, encoder, batch, keys, num_frames, args, share_map):
    """Per-metric dispatch — returns (per_example_features, per_example_labels) on CPU, where
    each clip contributes K variant examples (K=2 aot, K=n_permutations tov, K=len(strides)
    pace, K=1 tcc-per-frame stack). ALL layouts are CLIP-MAJOR (clip0's K rows contiguous) so
    (a) the per-clip collapse reshape(-1, K) pairs the right rows — the legacy aot layout was
    VARIANT-MAJOR ([fwd×B, rev×B]) which silently mis-paired per-clip correctness — and
    (b) OOM-halved sub-batches concatenate correctly (iter18 #4).
    share_map (iter18 #6, m12b pattern): clip_key → (D,) fp32 IDENTITY-forward features from
    probe_action's cache; when present, aot computes only the REVERSE forward and tov skips
    the identity permutation."""
    # iter19 2026-07-09 rung 3 (zero-idle): upload the pinned batch ONCE — every variant gather
    # below (aot flip · tov permute · pace stride index_select · tcc pass-through) then runs ON
    # GPU instead of single-thread CPU index-shuffling between forwards (measured post-rung-2:
    # tov idled 41% of wall, burning exactly 1 CPU core during each 0% window = the CPU gather).
    # Pure indexing/data-movement — values byte-identical; to_pixel's .to("cuda") becomes a no-op
    # move + bf16 cast. non_blocking: the BatchFeeder batch is pinned → async DMA.
    batch = batch.to("cuda", non_blocking=True)
    if metric == "aot":
        B = batch.shape[0]
        if share_map is not None:
            f_fwd = torch.stack([share_map[k] for k in keys])                      # (B, D) cached
            f_rev = et_aot.compute_features_rev(encoder, batch, num_frames, args.tubelet_size)
        else:
            f_fwd, f_rev = et_aot.compute_features(encoder, batch, num_frames, args.tubelet_size)
        feats = torch.stack([f_fwd, f_rev], dim=1).reshape(2 * B, -1)              # clip-major
        labels = torch.tensor([0, 1], dtype=torch.long).repeat(B)
        return feats, labels
    if metric == "tov":
        T = batch.shape[1]
        perms = _gen_permutations(args.tov_n_permutations, T, seed=args.seed)
        if share_map is not None:
            rest = et_tov.compute_features(encoder, batch, num_frames, args.tubelet_size,
                                           perms, skip_first=True)                 # (K-1, B, D)
            ident = torch.stack([share_map[k] for k in keys]).unsqueeze(0)          # (1, B, D)
            stack = torch.cat([ident, rest], dim=0)
        else:
            stack = et_tov.compute_features(encoder, batch, num_frames, args.tubelet_size,
                                            perms, skip_first=False)               # (K, B, D)
        n_perm, B, D = stack.shape
        feats = stack.permute(1, 0, 2).reshape(B * n_perm, D)                       # clip-major
        labels = torch.arange(n_perm).repeat(B)
        return feats, labels
    if metric == "pace":
        strides = [int(s) for s in args.pace_strides.split(",")]
        if sorted(strides) != strides or len(set(strides)) != len(strides):
            raise SystemExit(f"FATAL: --pace-strides must be sorted + distinct; got {strides}")
        stack = et_pace.compute_features(encoder, batch, num_frames, args.tubelet_size, strides)
        n_str, B, D = stack.shape
        feats = stack.permute(1, 0, 2).reshape(B * n_str, D)
        labels = torch.arange(n_str).repeat(B)
        return feats, labels
    if metric == "tcc":
        per_frame = et_tcc.compute_per_frame(encoder, batch, num_frames, args.tubelet_size)
        # TCC stores per-frame features (B, T_eff, D) ; "labels" carry no class meaning here
        # and are unused by compute_pair downstream. We use a sentinel -1 array.
        labels = -torch.ones(per_frame.shape[0], dtype=torch.long)
        return per_frame, labels  # NOTE: 3-D feats for tcc — handled by stage 2 dispatch
    raise SystemExit(f"FATAL: unknown metric {metric!r}")


_K_PER_CLIP = {"aot": lambda a: 2, "tov": lambda a: a.tov_n_permutations,
               "pace": lambda a: len(a.pace_strides.split(",")), "tcc": lambda a: 1}


def _resolve_tcc_pair_chunk(arg_val, t_eff, d, n_pairs):
    """'auto' → pairs per batched-TCC chunk derived from FREE VRAM × the universal
    gpu.gpu_memory_target cap (same source AdaptiveBatchSizer uses — scales itself across
    24/48/96 GB boxes); an explicit int pins it. Per-pair device bytes = the 3 (T_eff, D)
    pair tensors compute_pairs_batched holds (A, B, gathered sel_b) + 2 (T_eff, T_eff)
    sim/softmax maps, fp32, ×2 safety for kernel temporaries."""
    if str(arg_val).lower() != "auto":
        return max(1, int(arg_val))
    free_b, _total = torch.cuda.mem_get_info()
    target = float(_PCFG["gpu"]["gpu_memory_target"])
    per_pair = (3 * t_eff * d + 2 * t_eff * t_eff) * 4 * 2
    chunk = max(1, min(int(n_pairs), int(free_b * target / per_pair)))
    print(f"  [tcc] pair-chunk auto → {chunk}  "
          f"(free={free_b / 1e9:.1f}G × target={target} ÷ {per_pair / 1e6:.2f}MB/pair, "
          f"n_pairs={n_pairs})", flush=True)
    return chunk


def _forward_with_oom_backoff(metric, encoder, batch_t, keys, num_frames, args, share_map):
    """Run the metric forward on a pre-STACKED clip batch (B,T,3,H,W), HALVING on CUDA OOM down
    to 1 clip (iter18 #4 — mirrors utils.predictor_eval.safe_metric's intent; replaces the legacy
    FATAL exit). iter19 07-09: takes the already-stacked (pinned) tensor from BatchFeeder — the
    stack moved OFF the GPU loop onto the builder thread, and the halves are leading-dim SLICES
    (contiguous views, still pinned) so the retry re-stacks nothing; values byte-identical.
    The retry runs OUTSIDE the except block: an exception's __traceback__ pins the FAILED
    forward's activations, so recursing inside `except` accumulates every failed attempt's
    VRAM — observed 2026-06-12 smoke: aot's 40-sample forward fit, then tov OOM'd even at a
    4-sample forward after 4 nested retries. Leaving the except scope (and cuda_cleanup'ing)
    releases the pinned tensors before each retry."""
    oom = False
    try:
        return _run_metric_forward(metric, encoder, batch_t, keys,
                                   num_frames, args, share_map)
    except torch.cuda.OutOfMemoryError:
        if batch_t.shape[0] == 1:
            raise SystemExit(
                f"FATAL: OOM at a SINGLE-clip batch for metric={metric} — the stacked variant "
                f"forward does not fit this GPU even at B=1; lower --num-frames/--batch-size "
                f"or run on a larger card") from None
        oom = True
    # ── out of the except scope: traceback freed → the failed attempt's VRAM is reclaimable ──
    assert oom
    cuda_cleanup()
    mid = batch_t.shape[0] // 2
    print(f"  [oom-backoff] {metric}: batch {batch_t.shape[0]} → {mid}+{batch_t.shape[0] - mid}",
          flush=True)
    f1, l1 = _forward_with_oom_backoff(metric, encoder, batch_t[:mid], keys[:mid],
                                       num_frames, args, share_map)
    f2, l2 = _forward_with_oom_backoff(metric, encoder, batch_t[mid:], keys[mid:],
                                       num_frames, args, share_map)
    return torch.cat([f1, f2], dim=0), torch.cat([l1, l2], dim=0)


def _load_share_map(args, split_name, split_keys, embed_dim):
    """iter18 #6 (m12b --share-features pattern): mean-pool probe_action's cached
    features_{split}.npy → {clip_key: (D,) fp32} serving the IDENTITY forward (AoT-fwd /
    TOV perm-0). Pooling parity: probe_action stores forward_per_frame's (N, T_eff, D)
    spatial-means, so .mean(dim=1) here equals the .mean(dim=1) inside et_aot/et_tov.
    Returns (map, "ok") or (None, reason) — caller prints the decision and falls back to
    fresh identity forwards (explicit, never silent)."""
    if not args.share_features:
        return None, "disabled (--share-features 0)"
    enc_dir = args.action_probe_root / args.variant
    src, kf = enc_dir / f"features_{split_name}.npy", enc_dir / f"clip_keys_{split_name}.npy"
    if not (src.exists() and kf.exists()):
        return None, f"probe_action cache missing at {enc_dir}"
    feats = np.load(src)
    t_eff = args.num_frames // args.tubelet_size
    if feats.ndim != 3 or feats.shape[2] != embed_dim or feats.shape[1] != t_eff:
        return None, (f"cache shape {feats.shape} mismatches (N, T_eff={t_eff}, D={embed_dim}) "
                      f"— pooling parity not guaranteed")
    ckeys = [str(k) for k in np.load(kf, allow_pickle=True)]
    pooled = torch.from_numpy(feats.astype(np.float32)).mean(dim=1)        # (N, D)
    smap = {k: pooled[i] for i, k in enumerate(ckeys)}
    missing = sum(1 for k in split_keys if k not in smap)
    if missing:
        return None, f"{missing}/{len(split_keys)} split keys absent from cache"
    return smap, "ok"


def _extract_split_multi(args, encoder, metrics, split_keys, split_name, out_dir, decode_T,
                         share_map):
    """Stream the split through the encoder ONCE for ALL `metrics` that share this decode
    length (iter18 #7 — replaces per-metric streaming; pace runs its own OVERSAMPLE pass).
    Returns {metric: (feats, labels, expanded_clip_ids)} and persists the legacy per-metric
    npys. Crash-safe (iter18 #5): accumulated features checkpoint to .m12f_ckpt_{split}.npz
    every CHECKPOINT_EVERY clips; on restart the processed clips are skipped via
    iter_clips_parallel(processed_keys=...). FAIL LOUD if the resulting feature N is 0."""
    if not split_keys:
        raise SystemExit(f"FATAL: split={split_name} has 0 keys — pipeline failure")
    tmp_root = out_dir / f"tmp_decode_{'-'.join(metrics)}_{split_name}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_root)
    if "pace" in metrics:
        max_stride = max(int(s) for s in args.pace_strides.split(","))
        if decode_T < args.num_frames * max_stride:
            print(f"  WARN: --pace-source-frames {decode_T} < num_frames * max_stride "
                  f"({args.num_frames * max_stride}); stride_indices will cyclic-wrap")

    feats_all = {m: [] for m in metrics}
    labels_all = {m: [] for m in metrics}
    clip_ids_all = []
    # ── iter18 #5: resume from the intra-run checkpoint (same metric GROUP only) ──
    # iter19 2026-07-09 COLLISION FIX (caught live by /gpu-bottleneck): the ckpt name was split-only
    # (.m12f_ckpt_test.npz) but FOUR per-metric F: jobs share out_dir — the last writer won and every
    # sibling hit "holds metrics=['pace'] ≠ ['tov'] — discarding" on restart (tov re-extracted ~9.6k
    # test clips ≈ 1.6h GPU, twice). Now metric-qualified: .m12f_ckpt_<metrics>_<split>.npz — tov/tcc
    # can both run the test split concurrently without clobbering. LEGACY fallback: a pre-fix job
    # still writing the split-only name resumes IFF its stored metrics match (never discards a
    # sibling's work — mismatch on the legacy name just means "not ours", not "corrupt").
    ckpt = out_dir / f".m12f_ckpt_{'-'.join(metrics)}_{split_name}.npz"
    _legacy_ckpt = out_dir / f".m12f_ckpt_{split_name}.npz"
    _resume_src = ckpt if ckpt.exists() else (_legacy_ckpt if _legacy_ckpt.exists() else None)
    if _resume_src is not None:
        z = np.load(_resume_src, allow_pickle=True)
        stored = [str(m) for m in z["metrics"]]
        if stored == list(metrics):
            clip_ids_all = [str(k) for k in z["clip_ids"]]
            for m in metrics:
                feats_all[m] = [torch.from_numpy(z[f"feats_{m}"])]
                labels_all[m] = [torch.from_numpy(z[f"labels_{m}"])]
            print(f"  [resume] {_resume_src.name}: {len(clip_ids_all)} clips already extracted "
                  f"({split_name}, metrics={stored})")
        elif _resume_src is ckpt:
            print(f"  [resume] {ckpt.name} holds metrics={stored} ≠ current {list(metrics)} "
                  f"— discarding (full re-extract)")
        else:
            print(f"  [resume] legacy {_resume_src.name} holds metrics={stored} (another metric's "
                  f"work) — leaving it alone, fresh extract for {list(metrics)}")
    processed = set(clip_ids_all)

    # iter19 2026-07-09: ckpt save OFF the hot path — the inline torch.cat + np.savez of the whole
    # cumulative feature set (GBs, O(n) per save) stalled the GPU every CHECKPOINT_EVERY clips.
    # Snapshot = shallow list copies (tensors are append-only, never mutated) → ONE background
    # writer cats+savez's while the forwards continue; ≤1 outstanding (join before respawn); the
    # atomic os.replace publish is unchanged, so a crash mid-write never corrupts the ckpt.
    _ckpt_thread = [None]

    def _save_ckpt():
        snap_f = {m: list(feats_all[m]) for m in metrics}
        snap_l = {m: list(labels_all[m]) for m in metrics}
        snap_ids = list(clip_ids_all)
        if _ckpt_thread[0] is not None:
            _ckpt_thread[0].join()

        def _write():
            arrays = {"metrics": np.array(list(metrics), dtype=object),
                      "clip_ids": np.array(snap_ids, dtype=object)}
            for m in metrics:
                arrays[f"feats_{m}"] = torch.cat(snap_f[m], dim=0).numpy()
                arrays[f"labels_{m}"] = torch.cat(snap_l[m], dim=0).numpy()
            # tmp name metric-qualified too (07-09): concurrent sibling jobs on the SAME split
            # (tov/test ∥ tcc/test) would interleave writes into one shared tmp → corrupt savez.
            tmp = out_dir / f".m12f_ckpt_{'-'.join(metrics)}_{split_name}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, ckpt)

        _ckpt_thread[0] = threading.Thread(target=_write, daemon=True)
        _ckpt_thread[0].start()                                # atomic publish

    clip_q, tar_stop, _r = iter_clips_parallel(
        local_data=args.local_data, subset_keys=set(split_keys), processed_keys=processed)
    since_ckpt = 0
    pbar = make_pbar(total=len(split_keys), desc=f"m12f[{'-'.join(metrics)}/{split_name}]",
                     unit="clip")
    pbar.update(len(processed))

    # iter19 2026-07-09: zero-idle, two rungs (utils/decode_feeder — SHARED util).
    # Rung 1 DecodeFeeder: N workers own clip_q→decode→ready_q, decode never pauses (the 07-08
    #   submit-refill pool refilled only between future-pops → workers idled during every forward;
    #   measured: pace 1.3-1.4 → 0.5-0.6 s/clip).
    # Rung 2 BatchFeeder: a builder thread torch.stack's + PINS the next batch while the current
    #   one forwards (double-buffer, depth 2) — the ~1-2s single-thread stack of a 3.4G pace batch
    #   used to run on the GPU loop between flushes = the residual 0% troughs at the ~20s batch
    #   cadence. Pixels/values BYTE-IDENTICAL: same decode, same stack, same order.
    feeder = DecodeFeeder(
        clip_q,
        decode_one=lambda ck, mp4: decode_to_tensor(mp4, tmp_dir, ck, decode_T, CROP),
        n_workers=max(1, args.decode_workers), ready_depth=args.batch_size * 3, timeout_s=300)
    batches = BatchFeeder(feeder, batch_size=args.batch_size, depth=2, pin_memory=True)
    try:
        for pend_k, batch_t in batches:
            for m in metrics:
                feats, labels = _forward_with_oom_backoff(m, encoder, batch_t, pend_k,
                                                          args.num_frames, args, share_map
                                                          if m in ("aot", "tov") else None)
                feats_all[m].append(feats)
                labels_all[m].append(labels)
            clip_ids_all.extend(pend_k)
            since_ckpt += len(pend_k)
            pbar.update(len(pend_k))
            if since_ckpt >= CHECKPOINT_EVERY:
                _save_ckpt()
                since_ckpt = 0
    finally:
        if _ckpt_thread[0] is not None:
            _ckpt_thread[0].join()          # let the last background ckpt writer land (atomic publish)
        tar_stop.set()
        pbar.close()

    out = {}
    for m in metrics:
        if not feats_all[m]:
            raise SystemExit(f"FATAL: {m}/{split_name} produced 0 features — pipeline failure")
        feats_cat = torch.cat(feats_all[m], dim=0)
        labels_cat = torch.cat(labels_all[m], dim=0)
        k = _K_PER_CLIP[m](args)
        ids_expanded = [ck for ck in clip_ids_all for _ in range(k)]
        np.save(out_dir / f"{m}_{split_name}_feats.npy", feats_cat.cpu().numpy())
        np.save(out_dir / f"{m}_{split_name}_labels.npy", labels_cat.cpu().numpy())
        np.save(out_dir / f"{m}_{split_name}_clip_ids.npy",
                np.array(ids_expanded, dtype=object))
        out[m] = (feats_cat, labels_cat, ids_expanded)
    return out


def _train_eval_classifier(metric, train_feats, train_labels, test_feats, test_labels, args, embed_dim, out_dir):
    """Trainable-head metrics (aot/tov/pace): train head on train split, evaluate per-example
    on test split, return per-test-example correctness. iter18 cross-set: --head-ckpt-dir loads the
    eval_10k head + skips training (test-all has train=0); --keep-heads saves the fresh head for reuse."""
    device = "cuda"
    # per-metric, NOT a dict literal — a dict eagerly evaluates pace_strides.split() even for
    # aot/tov (where --pace-strides is legitimately None) → AttributeError.
    if metric == "aot":
        n_classes = 2
    elif metric == "tov":
        n_classes = args.tov_n_permutations
    elif metric == "pace":
        n_classes = len(args.pace_strides.split(","))
    else:
        raise SystemExit(f"FATAL: _train_eval_classifier got non-head metric {metric!r}")
    mod = {"aot": et_aot, "tov": et_tov, "pace": et_pace}[metric]
    # iter18 head-reuse (Stage B): load the eval_10k head (nn.Linear), skip training, eval TEST only.
    if args.head_ckpt_dir is not None:
        head_path = args.head_ckpt_dir / f"head_{metric}.pt"
        if not head_path.exists():
            raise SystemExit(f"FATAL: --head-ckpt-dir head {head_path} missing — run the eval_10k gen with --keep-heads first")
        head = torch.nn.Linear(embed_dim, n_classes)
        head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        head = head.to(device)
        print(f"[head-reuse] {metric}: loaded {head_path} → Linear({embed_dim},{n_classes}); training SKIPPED")
        return mod.eval_head(head, test_feats, test_labels, device=device, batch_size=args.head_batch_size)
    kwargs = dict(embed_dim=embed_dim, lr=args.head_lr, epochs=args.head_epochs,
                  weight_decay=args.head_weight_decay, batch_size=args.head_batch_size,
                  device=device, seed=args.seed)
    if metric == "aot":
        head = mod.train_head(train_feats, train_labels, **kwargs)
    else:
        head = mod.train_head(train_feats, train_labels, n_classes=n_classes, **kwargs)
    if args.keep_heads:   # save the fresh head as the cross-set reuse SOURCE (mirrors m12c probe_<dim>.pt)
        torch.save(head.state_dict(), out_dir / f"head_{metric}.pt")
        print(f"[keep-heads] {metric}: saved head → {out_dir / f'head_{metric}.pt'}")
    return mod.eval_head(head, test_feats, test_labels, device=device, batch_size=args.head_batch_size)


def _tcc_scores(test_feats_per_frame, test_clip_ids, args):
    """Training-free TCC scoring: pair test clips by action label, average cycle-back error
    and Kendall's τ per clip across its pairs.
    iter18 #8: the per-pair python loop (O(P²) compute_pair calls) is replaced by
    et_tcc.compute_pairs_batched — all pairs' T×T soft-NN chains run as chunked batched
    matmuls on the GPU (--tcc-pair-chunk bounds the (chunk, T, D) gather memory); per-clip
    accumulation is vectorized with np.add.at. Same math (parity-tested vs compute_pair,
    scipy τ-b included); NaN-τ degenerate pairs excluded from the τ mean exactly as before."""
    labels_map = load_action_labels(args.action_probe_root / "action_labels.json")
    by_action = {}
    for i, ck in enumerate(test_clip_ids):
        a = labels_map[ck]["class_id"]  # action_labels schema: {class, class_id, split}
        by_action.setdefault(a, []).append(i)
    # iter19 2026-07-09 (CPU→GPU audit): the enumeration below was an O(P) python triple-loop — ~70M
    # list.append calls (~10s single-thread at 0% GPU, bracketing the already-GPU pairwise matmul).
    # Vectorized per action group with np.triu_indices(k=1), which yields the IDENTICAL (ii<jj) row-major
    # order as the double loop → pairs_a/pairs_b byte-identical, so the chunked gather + np.add.at means
    # downstream are unchanged (smoke-verified array-equal). Pure index construction; no fp math.
    _pa_parts, _pb_parts = [], []
    for _a, idxs in by_action.items():
        if len(idxs) < 2:
            continue
        _idx = np.asarray(idxs)
        _ii, _jj = np.triu_indices(len(idxs), k=1)
        _pa_parts.append(_idx[_ii]); _pb_parts.append(_idx[_jj])
    pairs_a = np.concatenate(_pa_parts) if _pa_parts else np.empty(0, dtype=np.intp)
    pairs_b = np.concatenate(_pb_parts) if _pb_parts else np.empty(0, dtype=np.intp)
    n_pairs = len(pairs_a)
    if n_pairs == 0:
        raise SystemExit("FATAL: TCC produced 0 pairs — every action class has <2 test clips")
    feats = (test_feats_per_frame if torch.is_tensor(test_feats_per_frame)
             else torch.from_numpy(test_feats_per_frame)).float()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # iter19 2026-07-09 STUCK-FIX (/gpu-bottleneck): the loop gathered feats[pa[s:e]] ON CPU then .to(dev)
    # — at chunk=70547 a ~15GB single-thread host advanced-index gather PER SIDE (30GB/chunk) + a 30GB
    # host→device copy → GPU0 hung at 0% / 1-core-100% / 25GB RSS, NO progress for >1h over 494 chunks.
    # Fix = utils.gpu_gather.device_gather: upload feats ONCE (~5GB) + gather each chunk ON-DEVICE
    # (index_select). Byte-identical; feats on-GPU BEFORE the chunk-size heuristic so free_b accounts for
    # it; + a progress bar so it can never be a silent stall again.
    feats = feats.to(dev)
    # pa/pb stay NUMPY: device_gather converts each chunk slice pa[s:e] to a GPU long tensor internally
    # (a ~0.5MB H2D per chunk, negligible), AND the per-clip np.add.at accumulation below (~L499-504)
    # requires numpy indices — making pa/pb CUDA tensors crashed np.add.at ("can't convert cuda tensor
    # to numpy"): the pairwise loop ran (34.8M pairs/95s) but the aggregation right after it aborted.
    pa, pb = np.asarray(pairs_a), np.asarray(pairs_b)
    cyc_all = np.empty(n_pairs, dtype=np.float64)
    tau_all = np.empty(n_pairs, dtype=np.float64)
    chunk = _resolve_tcc_pair_chunk(args.tcc_pair_chunk, feats.shape[1], feats.shape[2], n_pairs)
    pbar = make_pbar(total=n_pairs, desc="m12f[tcc-pairs/test]", unit="pair")
    for s in range(0, n_pairs, chunk):
        e = min(s + chunk, n_pairs)
        c, t = et_tcc.compute_pairs_batched(device_gather(feats, pa[s:e]),
                                            device_gather(feats, pb[s:e]),
                                            temperature=args.tcc_temperature)
        cyc_all[s:e] = c.cpu().numpy()
        tau_all[s:e] = t.cpu().numpy()
        pbar.update(e - s)
    pbar.close()
    n_degenerate = int(np.isnan(tau_all).sum())
    # Kendall's τ is NaN for a degenerate pair (collinear per-frame features — expected for static
    # clips, esp. frozen encoders). The τ mean excludes those pairs so one bad pair does not abort
    # the eval; report the count so the degeneracy stays VISIBLE (FAIL-LOUD) rather than swallowed.
    if n_degenerate:
        print(f"  [tcc] {n_degenerate}/{n_pairs} pairs degenerate (collinear features) → τ=NaN, "
              f"excluded from per-clip τ mean (cycle-back kept).", flush=True)
    n_clips = feats.shape[0]
    cyc_sum, cyc_cnt = np.zeros(n_clips), np.zeros(n_clips)
    np.add.at(cyc_sum, pa, cyc_all); np.add.at(cyc_sum, pb, cyc_all)
    np.add.at(cyc_cnt, pa, 1);       np.add.at(cyc_cnt, pb, 1)
    val = ~np.isnan(tau_all)
    tau_sum, tau_cnt = np.zeros(n_clips), np.zeros(n_clips)
    np.add.at(tau_sum, pa[val], tau_all[val]); np.add.at(tau_sum, pb[val], tau_all[val])
    np.add.at(tau_cnt, pa[val], 1);            np.add.at(tau_cnt, pb[val], 1)
    cycle_arr = np.where(cyc_cnt > 0, cyc_sum / np.maximum(cyc_cnt, 1), np.nan).astype(np.float32)
    tau_arr = np.where(tau_cnt > 0, tau_sum / np.maximum(tau_cnt, 1), np.nan).astype(np.float32)
    keep = ~(np.isnan(cycle_arr) | np.isnan(tau_arr))
    if not keep.any():
        raise SystemExit("FATAL: TCC every test clip lacked a same-action pair")
    return cycle_arr[keep], tau_arr[keep], int(n_pairs), [test_clip_ids[i] for i in range(len(keep)) if keep[i]]


def _load_tcc_perframe_cache(args, test_keys):
    """OPT-IN (iter19): load probe_action's per-tubelet fp32 cache (perframe_test_fp32.npy +
    clip_keys_test.npy) → (feats (N,T_eff,D) fp32 tensor, sentinel labels, ordered clip_ids), so tcc
    SKIPS its 23k-clip re-encode. Byte-identical to a fresh compute_per_frame (same encoder forward,
    same CUDA per-tubelet pool, fp32). FAIL LOUD — never silently confound: missing cache, fp16 dtype,
    wrong rank, or incomplete test coverage all raise (a fresh re-encode is the safe fallback, so drop
    the flag to get it)."""
    root = args.tcc_perframe_cache_root / args.variant
    pf_path, key_path = root / "perframe_test_fp32.npy", root / "clip_keys_test.npy"
    if not (pf_path.exists() and key_path.exists()):
        raise SystemExit(f"FATAL: --tcc-perframe-cache-root set but cache missing at {root} (need "
                         f"perframe_test_fp32.npy + clip_keys_test.npy). Run probe_action Stage 2 with "
                         f"--emit-perframe-cache first, or drop the flag to re-encode.")
    feats = np.load(pf_path)
    if feats.dtype != np.float32:
        raise SystemExit(f"FATAL: {pf_path.name} dtype={feats.dtype} — tcc perframe cache MUST be fp32 "
                         f"(fp16 would silently confound the metric). Re-emit with --emit-perframe-cache.")
    if feats.ndim != 3:
        raise SystemExit(f"FATAL: {pf_path.name} shape {feats.shape} — expected (N, T_eff, D) per-tubelet.")
    ckeys = [str(k) for k in np.load(key_path, allow_pickle=True)]
    if len(ckeys) != feats.shape[0]:
        raise SystemExit(f"FATAL: perframe key/feature count mismatch ({len(ckeys)} vs {feats.shape[0]}).")
    want = set(test_keys)
    missing = want - set(ckeys)
    if missing:
        raise SystemExit(f"FATAL: perframe cache missing {len(missing)}/{len(want)} test clips "
                         f"(e.g. {sorted(missing)[:3]}) — stale cache; re-emit or drop the flag.")
    keep = [i for i, k in enumerate(ckeys) if k in want]
    feats_t = torch.from_numpy(feats[keep]).float()
    ids = [ckeys[i] for i in keep]
    labels = -torch.ones(feats_t.shape[0], dtype=torch.long)     # tcc labels are unused sentinels
    print(f"  [tcc-cache] loaded {tuple(feats_t.shape)} fp32 per-tubelet features from {pf_path} — "
          f"SKIPPING the ~23k-clip re-encode (byte-identical to a fresh forward_per_frame)", flush=True)
    return feats_t, labels, ids


def run_forward_stage(args, wb):
    for need, msg in [(args.variant, "--variant"), (args.encoder_ckpt, "--encoder-ckpt"),
                      (args.local_data, "--local-data"),
                      (args.action_probe_root, "--action-probe-root")]:
        if need is None:
            raise SystemExit(f"FATAL: --stage forward requires {msg}")
    if not args.variant.startswith("vjepa"):
        raise SystemExit(
            f"FATAL: m12f forward only supports vjepa_* variants (encoder-temporal "
            f"needs a video encoder); got {args.variant!r}")
    metrics = _resolve_metrics(args.metric)

    check_gpu()
    print_cgroup_header(prefix="[m12f_encoder_temporal]")
    start_oom_watchdog(prefix="[m12f_encoder_temporal]-oom-watchdog")
    cleanup_temp()
    ensure_local_data(args)

    out_dir = args.output_root / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for m in metrics:
        agg = out_dir / f"aggregate_{m}.json"
        if agg.exists() and args.cache_policy == "1":
            print(f"  [keep] {m}: {agg.name} present — skipping (--cache-policy 2 to redo)")
            continue
        for p in out_dir.glob(f"per_clip_{m}*.npy"):
            guarded_delete(p, args.cache_policy, p.name)
        guarded_delete(agg, args.cache_policy, agg.name)
        todo.append(m)
    if not todo:
        print("  all requested metrics cached — nothing to do")
        return
    # iter18 #5: a full recompute must not RESUME from a prior run's intra-run checkpoint —
    # clear the ckpts under policy 2 (policy 1 keeps them = crash-resume). 07-09: covers both the
    # metric-qualified names (glob) and the legacy split-only name.
    if args.cache_policy == "2":
        for sp in ("test", "train"):
            for p in out_dir.glob(f".m12f_ckpt_*{sp}.npz"):
                guarded_delete(p, args.cache_policy, p.name)

    labels_map = load_action_labels(args.action_probe_root / "action_labels.json")
    train_keys = sorted(k for k, info in labels_map.items() if info["split"] == "train")
    test_keys = sorted(k for k, info in labels_map.items() if info["split"] == "test")
    # §opt: cap head-fit TRAIN clips (linear head saturates well before full N); TEST + BCa CI untouched.
    if args.head_train_cap and len(train_keys) > args.head_train_cap:
        print(f"  [head-train-cap] train {len(train_keys)} → {args.head_train_cap} (head-fit only; test/CI unaffected)")
        train_keys = train_keys[:args.head_train_cap]
    print(f"  splits: train={len(train_keys)}  test={len(test_keys)}  metrics={todo}")

    encoder, _ckpt, embed_concat = load_encoder_only(args.encoder_ckpt, args.num_frames, args.model_config)
    del _ckpt  # encoder-temporal needs no predictor (iter16 §3.3 R2 — no predictor build)
    print(f"  encoder loaded (concat dim={embed_concat}; encoder-only — no predictor build)")

    # ── iter18 #7: group metrics by decode length — aot/tov/tcc share the num_frames decode →
    # ONE streaming pass per split runs ALL their forwards on the same decoded batch (was: one
    # full decode+stream per metric). pace keeps its own OVERSAMPLE pass (different decode_T;
    # slicing a longer decode down would NOT be bit-identical to decoding at num_frames —
    # ACCURACY OVER SPEED). The frame cache (EVAL_FRAME_CACHE_DIR) additionally memoizes the
    # decode across processes/runs per (clip, decode_T). ──
    # iter19 opt-in tcc-cache: read probe_action's fp32 per-tubelet cache instead of re-encoding tcc.
    _tcc_cached = None
    if "tcc" in todo and getattr(args, "tcc_perframe_cache_root", None) is not None:
        _tcc_cached = _load_tcc_perframe_cache(args, test_keys)
    groups = []
    std = [m for m in todo if m != "pace" and not (m == "tcc" and _tcc_cached is not None)]
    if std:
        groups.append((args.num_frames, std))
    if "pace" in todo:
        groups.append((args.pace_source_frames, ["pace"]))
    by_metric = {}
    if _tcc_cached is not None:
        by_metric["tcc"] = {"test": _tcc_cached, "train": None}
    for decode_T, ms in groups:
        share_eligible = any(m in ("aot", "tov") for m in ms)
        smap_test, why_t = (_load_share_map(args, "test", test_keys, embed_concat)
                            if share_eligible else (None, "n/a (no aot/tov in group)"))
        print(f"  [share-features] test: {'ON' if smap_test else f'off — {why_t}'}")
        test_res = _extract_split_multi(args, encoder, ms, test_keys, "test", out_dir,
                                        decode_T, smap_test)
        head_ms = [m for m in ms if m in ("aot", "tov", "pace")]
        train_res = {}
        if head_ms and args.head_ckpt_dir is None:   # head-reuse skips training → no train split needed (test-all has train=0)
            smap_tr, why_tr = (_load_share_map(args, "train", train_keys, embed_concat)
                               if share_eligible else (None, "n/a (no aot/tov in group)"))
            print(f"  [share-features] train: {'ON' if smap_tr else f'off — {why_tr}'}")
            train_res = _extract_split_multi(args, encoder, head_ms, train_keys, "train",
                                             out_dir, decode_T, smap_tr)
        for m in ms:
            by_metric[m] = {"test": test_res[m], "train": train_res.get(m)}

    for m in todo:
        t0 = time.time()
        test_feats, test_labels, test_ids = by_metric[m]["test"]
        if m in ("aot", "tov", "pace"):
            _tr = by_metric[m]["train"]   # None under head-reuse (train split not extracted)
            tr_feats, tr_labels = (_tr[0], _tr[1]) if _tr is not None else (None, None)
            per_example = _train_eval_classifier(m, tr_feats, tr_labels, test_feats, test_labels,
                                                 args, embed_dim=embed_concat, out_dir=out_dir)
            # collapse per-example (K per clip) to per-clip mean
            k = per_example.size // len(test_keys)
            per_clip = per_example.reshape(-1, k).mean(axis=1) if k > 1 else per_example
            save_array_checkpoint(per_clip, out_dir / f"per_clip_{m}.npy")
            ci = bootstrap_ci(per_clip.astype(np.float64))
            save_json_checkpoint({
                "variant": args.variant, "metric": m, "n_test": int(per_clip.size),
                "k_per_clip": int(k),
                "mean": round(float(per_clip.mean()), 6),
                "std": round(float(per_clip.std()), 6),
                "ci": ci, "lower_is_better": METRICS[m][0], "interpretation": METRICS[m][1],
                "num_frames": args.num_frames, "tubelet_size": args.tubelet_size,
                "encoder_ckpt": str(args.encoder_ckpt),
                "head_lr": args.head_lr, "head_epochs": args.head_epochs,
                "head_weight_decay": args.head_weight_decay,
                "head_batch_size": args.head_batch_size,
                **({"n_permutations": args.tov_n_permutations} if m == "tov" else {}),
                **({"strides": args.pace_strides, "source_frames": args.pace_source_frames}
                   if m == "pace" else {}),
            }, out_dir / f"aggregate_{m}.json")
            print(f"  {m}: mean={per_clip.mean():.4f} std={per_clip.std():.4f} "
                  f"N={per_clip.size} ({time.time()-t0:.0f}s)")
            log_metrics(wb, {f"{args.variant}_{m}_mean": float(per_clip.mean())})
        elif m == "tcc":
            cyc, tau, n_pairs, kept_ids = _tcc_scores(test_feats, test_ids, args)
            save_array_checkpoint(cyc, out_dir / "per_clip_tcc_cycle.npy")
            save_array_checkpoint(tau, out_dir / "per_clip_tcc_tau.npy")
            np.save(out_dir / "per_clip_tcc_clip_ids.npy", np.array(kept_ids, dtype=object))
            ci_cyc = bootstrap_ci(cyc.astype(np.float64))
            ci_tau = bootstrap_ci(tau.astype(np.float64))
            save_json_checkpoint({
                "variant": args.variant, "metric": "tcc",
                "n_test": int(cyc.size), "n_pairs": int(n_pairs),
                "cycle_back": {"mean": round(float(cyc.mean()), 6),
                               "std": round(float(cyc.std()), 6),
                               "ci": ci_cyc, "lower_is_better": True},
                "kendalls_tau": {"mean": round(float(tau.mean()), 6),
                                 "std": round(float(tau.std()), 6),
                                 "ci": ci_tau, "lower_is_better": False},
                "temperature": args.tcc_temperature,
                "num_frames": args.num_frames, "tubelet_size": args.tubelet_size,
                "encoder_ckpt": str(args.encoder_ckpt),
            }, out_dir / "aggregate_tcc.json")
            print(f"  tcc: cycle={cyc.mean():.4f}±{cyc.std():.4f}  τ={tau.mean():.4f}±{tau.std():.4f} "
                  f"N={cyc.size} pairs={n_pairs} ({time.time()-t0:.0f}s)")
            log_metrics(wb, {f"{args.variant}_tcc_cycle_mean": float(cyc.mean()),
                             f"{args.variant}_tcc_tau_mean": float(tau.mean())})


def run_paired_per_variant_stage(args, wb):
    """Pairwise BCa Δ across discovered vjepa variants, per metric."""
    metrics = _resolve_metrics(args.metric)
    # iter19 2026-07-09: PASS 1 builds every (metric, pair[, tcc-field]) delta; ONE parallel_bca fans all
    # the 10K-resample BCa across cores (byte-identical — fixed per-unit seed); PASS 2 assembles. Same
    # schema/order/values (metric order, i<j pairs, tcc's cycle_back/kendalls_tau split, lower_is_better).
    out = {"metrics": {}}
    _units, _slots, _meta = [], [], {}
    for m in metrics:
        by_variant = {}
        if m == "tcc":
            for v in KNOWN_VARIANTS:
                cyc = args.output_root / v / "per_clip_tcc_cycle.npy"
                tau = args.output_root / v / "per_clip_tcc_tau.npy"
                ids = args.output_root / v / "per_clip_tcc_clip_ids.npy"
                agg = args.output_root / v / "aggregate_tcc.json"
                if not (cyc.exists() and tau.exists() and ids.exists() and agg.exists()):
                    continue
                by_variant[v] = {
                    "agg": json.loads(agg.read_text()),
                    "cycle": np.load(cyc).astype(np.float64),
                    "tau": np.load(tau).astype(np.float64),
                    "keys": [str(k) for k in np.load(ids, allow_pickle=True)],
                }
            avail = sorted(by_variant)
            _meta["tcc"] = ({v: by_variant[v]["agg"] for v in avail}, True, None)
            for i, a in enumerate(avail):
                for b in avail[i + 1:]:
                    ka, kb = by_variant[a]["keys"], by_variant[b]["keys"]
                    shared = sorted(set(ka) & set(kb))
                    if not shared:
                        raise SystemExit(f"FATAL [tcc]: {a} vs {b} share 0 clips")
                    ai = {k: j for j, k in enumerate(ka)}
                    bi = {k: j for j, k in enumerate(kb)}
                    for fld in ("cycle", "tau"):
                        d = (np.array([by_variant[a][fld][ai[k]] for k in shared])
                             - np.array([by_variant[b][fld][bi[k]] for k in shared]))
                        outfld = "cycle_back" if fld == "cycle" else "kendalls_tau"
                        _units.append(d)
                        _slots.append(("tcc", outfld, f"{a}_minus_{b}", len(shared), round(float(d.mean()), 6)))
            continue
        # trainable-head metrics
        for v in KNOWN_VARIANTS:
            pcl = args.output_root / v / f"per_clip_{m}.npy"
            agg = args.output_root / v / f"aggregate_{m}.json"
            ids = args.output_root / v / f"{m}_test_clip_ids.npy"
            if not (pcl.exists() and agg.exists() and ids.exists()):
                continue
            raw_ids = [str(k) for k in np.load(ids, allow_pickle=True)]
            # ids file contains K-per-clip duplicates ; collapse to unique in order.
            seen, dedup = set(), []
            for k in raw_ids:
                if k not in seen:
                    seen.add(k); dedup.append(k)
            by_variant[v] = {
                "agg": json.loads(agg.read_text()),
                "vals": np.load(pcl).astype(np.float64),
                "keys": dedup,
            }
        avail = sorted(by_variant)
        _meta[m] = ({v: by_variant[v]["agg"] for v in avail}, False, METRICS[m][0])
        for i, a in enumerate(avail):
            for b in avail[i + 1:]:
                ka, kb = by_variant[a]["keys"], by_variant[b]["keys"]
                shared = sorted(set(ka) & set(kb))
                if not shared:
                    raise SystemExit(f"FATAL [{m}]: {a} vs {b} share 0 clips")
                ai = {k: j for j, k in enumerate(ka)}
                bi = {k: j for j, k in enumerate(kb)}
                d = (np.array([by_variant[a]["vals"][ai[k]] for k in shared])
                     - np.array([by_variant[b]["vals"][bi[k]] for k in shared]))
                _units.append(d)
                _slots.append((m, None, f"{a}_minus_{b}", len(shared), round(float(d.mean()), 6)))
    _bcas = parallel_bca(_units)   # ONE fan-out; byte-identical to per-unit paired_bca(d)
    _pd = {}
    for (m, outfld, pkey, n, dmean), bca in zip(_slots, _bcas):
        entry = {
            "n": n, "delta_mean": dmean,
            "delta_ci_lo": round(float(bca["ci_lo"]), 6),
            "delta_ci_hi": round(float(bca["ci_hi"]), 6),
            "p_value": float(bca["p_value_vs_zero"]),
        }
        if outfld is not None:   # tcc → nested cycle_back / kendalls_tau
            _pd.setdefault(m, {"cycle_back": {}, "kendalls_tau": {}})[outfld][pkey] = entry
        else:
            _pd.setdefault(m, {})[pkey] = entry
    for m, (by_variant_agg, is_tcc, lib) in _meta.items():
        node = {"by_variant": by_variant_agg,
                "pairwise_deltas": _pd.get(m, {"cycle_back": {}, "kendalls_tau": {}} if is_tcc else {})}
        if not is_tcc:
            node["lower_is_better"] = lib
        out["metrics"][m] = node
    save_json_checkpoint(out, args.output_root / "encoder_temporal_per_variant.json")
    log_metrics(wb, {"n_metrics": len(out["metrics"])})
    print(json.dumps(out, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="m12f encoder-temporal metric suite (TOV/AoT/Pace/TCC)")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC", action="store_true")
    p.add_argument("--FULL", action="store_true")
    p.add_argument("--stage", required=True, choices=["forward", "paired_per_variant"])
    p.add_argument("--metric", required=True, choices=list(METRICS) + ["all"])
    p.add_argument("--variant", choices=list(KNOWN_VARIANTS), default=None)
    p.add_argument("--encoder-ckpt", type=Path, default=None)
    p.add_argument("--model-config", type=str, default=None,
                   help="configs/model/<backbone>.yaml — encoder arch/dims (WS-B3 arch-aware). "
                        "Required for --stage forward; unused by --stage paired_per_variant (None → ViT-G).")
    add_local_data_arg(p)
    p.add_argument("--action-probe-root", type=Path, default=None,
                   help="probe_action output dir (action_labels.json train/test splits)")
    p.add_argument("--tcc-perframe-cache-root", type=Path, default=None,
                   help="OPT-IN (iter19): probe_action's per-encoder features dir. When set + --metric tcc, "
                        "tcc reads <root>/<variant>/perframe_test_fp32.npy (emitted by probe_action "
                        "--emit-perframe-cache) instead of re-encoding — byte-identical, saves ~23k-clip "
                        "forward. FAIL LOUD if the fp32 cache is missing/wrong. Default None = re-encode.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES_DEFAULT)
    p.add_argument("--tubelet-size", type=int, required=True,
                   help="encoder tubelet size (V-JEPA 2.1 ViT-G = 2); per model-config")
    p.add_argument("--batch-size", type=int, default=4,
                   help="clips per encoder forward (OOM-protected; lower if Pace OOMs)")
    p.add_argument("--decode-workers", type=int,
                   default=_PCFG["probe"]["encoder_temporal"]["decode_workers"],
                   help="parallel clip-decode threads feeding the forward loop (iter19: pace's uncacheable "
                        "64f decode starved the GPU serially). Default: pipeline.yaml "
                        "probe.encoder_temporal.decode_workers.")
    p.add_argument("--seed", type=int, default=_PCFG["probe"]["seed"])
    # — per-metric sweep params (NO defaults — must come from pipeline.yaml at §3.3) —
    p.add_argument("--tov-n-permutations", type=int, required=False,
                   help="REQUIRED if --metric tov|all : number of TOV permutation classes (incl identity)")
    p.add_argument("--pace-strides", type=str, required=False,
                   help="REQUIRED if --metric pace|all : comma-separated sorted strides, e.g. 1,2,4")
    p.add_argument("--pace-source-frames", type=int, required=False,
                   help="REQUIRED if --metric pace|all : decode T_src; should be num_frames × max(strides)")
    p.add_argument("--tcc-temperature", type=float, required=False,
                   help="REQUIRED if --metric tcc|all : soft-NN temperature (Dwibedi eq.1)")
    p.add_argument("--tcc-pair-chunk", type=str, required=False,
                   help="REQUIRED if --metric tcc|all : pairs per batched TCC chunk — bounds the "
                        "(chunk, T_eff, D) gather memory of compute_pairs_batched (iter18 #8). "
                        "'auto' = derive from free VRAM × gpu.gpu_memory_target (scales across "
                        "24/48/96 GB boxes); an explicit int pins it. "
                        "yaml probe.encoder_temporal.tcc_pair_chunk")
    p.add_argument("--share-features", type=int, choices=[0, 1], required=False,
                   help="REQUIRED for --stage forward : 1 = serve the IDENTITY forward (AoT-fwd / "
                        "TOV perm-0) from probe_action's features_{split}.npy cache (m12b pattern; "
                        "iter18 #6) — explicit fallback to fresh forwards if the cache can't serve "
                        "every key ; 0 = always compute. yaml probe.encoder_temporal.share_features")
    # — head training (trainable metrics) —
    p.add_argument("--head-lr", type=float, required=False,
                   help="REQUIRED for aot|tov|pace|all : tiny linear head AdamW LR")
    p.add_argument("--head-epochs", type=int, required=False,
                   help="REQUIRED for aot|tov|pace|all : head training epochs")
    p.add_argument("--head-weight-decay", type=float, required=False,
                   help="REQUIRED for aot|tov|pace|all : head AdamW weight decay")
    p.add_argument("--head-batch-size", type=int, required=False,
                   help="REQUIRED for aot|tov|pace|all : head training batch size")
    p.add_argument("--head-train-cap", type=int, default=None,
                   help="Cap head-fit TRAIN clips to this many (linear head saturates ~1.5-2k). "
                        "TEST split + BCa CI are UNTOUCHED. None = full train. yaml encoder_temporal.head_train_cap.")
    # iter18 cross-set head-reuse (Stage B) — mirrors m12a --head-ckpt / m12c --head-ckpt-dir:
    p.add_argument("--keep-heads", action="store_true",
                   help="save each trained aot/tov/pace head to <out_dir>/head_<metric>.pt (the reuse SOURCE).")
    p.add_argument("--head-ckpt-dir", type=Path, default=None,
                   help="load per-metric head_<metric>.pt from here, SKIP training, eval TEST only "
                        "(pairs with PROBE_SPLIT=test-all where train=0).")
    add_cache_policy_arg(p)
    add_wandb_args(p)
    return p


def _check_required(args):
    """Stage-aware required-arg gate — argparse's required= can't condition on --metric, so we
    enforce here. FAIL LOUD on any missing knob for the requested metric."""
    if args.stage != "forward":
        return
    needs = {"tov": ["tov_n_permutations"],
             "pace": ["pace_strides", "pace_source_frames"],
             "tcc": ["tcc_temperature", "tcc_pair_chunk"]}
    head_needs = ["head_lr", "head_epochs", "head_weight_decay", "head_batch_size"]
    metrics = _resolve_metrics(args.metric)
    missing = []
    if args.share_features is None:
        missing.append("--share-features (stage=forward; 0|1, yaml encoder_temporal.share_features)")
    for m in metrics:
        for k in needs.get(m, []):
            if getattr(args, k) is None:
                missing.append(f"--{k.replace('_', '-')} (metric={m})")
        if m in ("aot", "tov", "pace"):
            for k in head_needs:
                if getattr(args, k) is None:
                    missing.append(f"--{k.replace('_', '-')} (metric={m})")
    if missing:
        raise SystemExit("FATAL: missing required args for --metric=" + args.metric
                         + ":\n  " + "\n  ".join(missing))


def main():
    args = build_parser().parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        raise SystemExit("ERROR: specify --SANITY, --POC, or --FULL")
    _check_required(args)
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)
    args.output_root.mkdir(parents=True, exist_ok=True)
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb = init_wandb(f"m12f_encoder_temporal_{args.stage}", mode,
                    config=vars(args), enabled=not args.no_wandb)
    try:
        torch.manual_seed(args.seed)
        if args.stage == "forward":
            t0 = time.time()
            run_forward_stage(args, wb)
            print(f"forward stage: {time.time() - t0:.0f}s")
        else:
            run_paired_per_variant_stage(args, wb)
    finally:
        finish_wandb(wb)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        print(f"\n❌ FATAL: {Path(__file__).name} crashed — see traceback below", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
