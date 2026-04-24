"""
Generate V-JEPA 2 embeddings via HF WebDataset streaming + producer-consumer GPU inference.
GPU-only (Nvidia CUDA required, no CPU fallback). Streams from HF — no local clips needed.

USAGE:
    python -u src/m05_vjepa_embed.py --SANITY 2>&1 | tee logs/m05_vjepa_embed_sanity.log
    python -u src/m05_vjepa_embed.py --POC --subset data/subset_10k.json --local-data data/subset_10k_local 2>&1 | tee logs/m05_vjepa_embed_poc.log
    python -u src/m05_vjepa_embed.py --FULL --local-data data/full_local 2>&1 | tee logs/m05_vjepa_embed_full.log
"""
import argparse
import contextlib
import errno
import gc
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    VJEPA_MODEL_ID, HF_DATASET_REPO, VJEPA_FRAMES_PER_CLIP,
    check_gpu, load_subset, add_subset_arg, add_local_data_arg, get_output_dir,
    get_module_output_dir,
    get_pipeline_config, get_sanity_clip_limit, get_total_clips,
    add_model_config_arg, get_model_config,
)
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.gpu_batch import compute_batch_sizes, add_gpu_mem_arg, cleanup_temp, AdaptiveBatchSizer
from utils.video_io import get_clip_key, create_stream, decode_video_bytes, _USE_TORCHCODEC
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, log_artifact, finish_wandb

try:
    import torch
    from transformers import AutoModel, AutoVideoProcessor
    from datasets import load_dataset  # noqa: F401 — imported for try/except check; used by utils.video_io
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install torch transformers datasets")
    sys.exit(1)

# torchcodec 0.11.0: SIGSEGV on decoder cleanup (del decoder triggers C++ destructor crash).
# Tested: decode works, but process dies on cleanup even with os._exit(0).
# Incompatible with PyTorch nightly 2.12+cu128 on Blackwell sm_120.
# Blocked until torchcodec fixes destructor bug. Using PyAV (CPU) as fallback.
# Constants
_pcfg = get_pipeline_config()
MAX_STREAM_RETRIES = _pcfg["streaming"]["max_retries"]
PREFETCH_QUEUE_SIZE = _pcfg["streaming"]["prefetch_queue_embed"]
CHECKPOINT_EVERY = _pcfg["streaming"]["checkpoint_every"]
DECODE_WORKERS = _pcfg["streaming"]["decode_workers_embed"]
ENGINE_RESTART_EVERY = _pcfg["streaming"]["engine_restart_every"]


# ── HF Streaming Helpers ──────────────────────────────────────────────
# Moved to utils/video_io.py: get_clip_key, create_stream, decode_video_bytes
# Backward compat alias (m05b/m05c may still import _create_stream from m05)
_create_stream = create_stream


# Video decoding moved to utils/video_io.py (decode_video_bytes, _load_av, _load_torchcodec)


# ── Checkpoint ─────────────────────────────────────────────────────────

def _checkpoint_fingerprint(model_path, is_adapted: bool) -> str:
    """Stable 8-char hash of (abs_path|size|mtime) for adapted ckpts; empty for frozen.
    Prevents cross-variant checkpoint collisions — prior to this, all 6 surgical
    variants shared `.m05_checkpoint_vjepa_2_1_surgical.npz`, so v11 would wrongly
    resume from v10's embeddings under v11's weights → garbage. iter10 2026-04-22.
    """
    if not is_adapted or model_path is None:
        return ""  # backward-compat: frozen filename unchanged
    try:
        p = Path(model_path).resolve()
        st = p.stat()
        return "_" + hashlib.sha256(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:8]
    except Exception:
        return "_nockpt"


def save_checkpoint(embeddings_list: list, keys_list: list, checkpoint_file: Path):
    """Atomic checkpoint save: embeddings + keys to .npz."""
    if not embeddings_list:
        return
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    # np.savez auto-appends .npz, so use _tmp.npz to get a predictable path
    tmp_file = checkpoint_file.with_name(checkpoint_file.stem + "_tmp.npz")
    np.savez(tmp_file,
             embeddings=np.stack(embeddings_list).astype(np.float32),
             keys=np.array(keys_list, dtype=object))
    os.replace(tmp_file, checkpoint_file)


def load_checkpoint(checkpoint_file: Path) -> tuple:
    """Load checkpoint. Returns (embeddings_list, keys_list, count)."""
    if not checkpoint_file.exists():
        return [], [], 0
    try:
        data = np.load(checkpoint_file, allow_pickle=True)
        emb_list = list(data["embeddings"])
        keys_list = list(data["keys"])
        print(f"Checkpoint loaded: {len(emb_list):,} embeddings from {checkpoint_file.name}")
        return emb_list, keys_list, len(emb_list)
    except Exception as e:
        print(f"  WARN: checkpoint corrupt ({e}), starting fresh")
        return [], [], 0


# ── Producer Thread (HF Stream → Decode → Queue) ──────────────────────

def _process_and_enqueue(processor, batch_tensors, batch_keys, q,
                         shuffle_frames: bool = False):
    """Process video tensors through processor and enqueue for GPU inference."""
    processed_list = []
    for vt, key in zip(batch_tensors, batch_keys):
        if shuffle_frames:
            # Deterministic shuffle per clip key (same as m05b shuffled baseline)
            seed = hash(key) % (2**31)
            rng = torch.Generator()
            rng.manual_seed(seed)
            perm = torch.randperm(vt.shape[0], generator=rng)
            vt = vt[perm]
        processed = processor(vt, return_tensors="pt")
        processed_list.append(processed["pixel_values_videos"])
    batched_pixels = torch.cat(processed_list, dim=0)
    q.put(("batch", batched_pixels, batch_keys[:]))


def _producer_thread(processor, batch_size: int, tmp_dir: str,
                     q: queue.Queue, stop_event: threading.Event,
                     clip_limit: int, subset_keys: set, num_frames: int,
                     processed_keys: set, local_data: str = None,
                     shuffle_frames: bool = False):
    """Stream from HF (or local shards), filter by subset, decode video tensors in parallel, enqueue."""
    produced = 0
    skipped = 0
    retries = 0
    tar_stop = None

    def _decode_and_enqueue(p_bytes, p_keys):
        """Parallel decode + process + enqueue one batch."""
        nonlocal produced
        with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
            futures = [
                pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
                for b, k in zip(p_bytes, p_keys)
            ]
            results = [(f.result(), k) for f, k in zip(futures, p_keys)]
        batch_tensors = [t for t, k in results if t is not None]
        batch_keys = [k for t, k in results if t is not None]
        if batch_tensors:
            _process_and_enqueue(processor, batch_tensors, batch_keys, q, shuffle_frames)
            produced += len(batch_tensors)

    try:
        pending_bytes = []
        pending_keys = []

        if local_data:
            # Fast path: parallel TAR readers (8 threads, skip processed keys at TAR level)
            clip_q, tar_stop, _reader = iter_clips_parallel(
                local_data, subset_keys=subset_keys, processed_keys=processed_keys)
            while produced < clip_limit and not stop_event.is_set():
                item = clip_q.get(timeout=120)
                if item is None:
                    break
                clip_key, mp4_bytes = item
                if not mp4_bytes:
                    continue
                pending_bytes.append(mp4_bytes)
                pending_keys.append(clip_key)
                if len(pending_bytes) >= batch_size:
                    _decode_and_enqueue(pending_bytes, pending_keys)
                    pending_bytes, pending_keys = [], []
                    if produced >= clip_limit:
                        break
        else:
            # Fallback: sequential HF streaming
            while produced < clip_limit and not stop_event.is_set():
                try:
                    ds = _create_stream(0, local_data=local_data)
                    for example in ds:
                        if stop_event.is_set():
                            break
                        clip_key = get_clip_key(example)
                        if subset_keys and clip_key not in subset_keys:
                            skipped += 1
                            continue
                        if clip_key in processed_keys:
                            continue
                        mp4_data = example.get("mp4", b"")
                        mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
                        if not mp4_bytes:
                            continue
                        pending_bytes.append(mp4_bytes)
                        pending_keys.append(clip_key)
                        if len(pending_bytes) >= batch_size:
                            _decode_and_enqueue(pending_bytes, pending_keys)
                            pending_bytes, pending_keys = [], []
                            retries = 0
                            if produced >= clip_limit:
                                break
                    break  # stream exhausted normally
                except (ConnectionError, TimeoutError, OSError) as e:
                    retries += 1
                    if retries > MAX_STREAM_RETRIES:
                        print(f"  ERROR: stream failed after {MAX_STREAM_RETRIES} retries: {e}")
                        break
                    wait = min(2 ** retries, 60)
                    print(f"  WARN: stream error ({e}), retry {retries}/{MAX_STREAM_RETRIES} in {wait}s")
                    time.sleep(wait)

        # Flush remaining partial batch
        if pending_bytes and not stop_event.is_set():
            _decode_and_enqueue(pending_bytes, pending_keys)

    except Exception as e:
        print(f"  ERROR: unexpected producer error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tar_stop:
            tar_stop.set()

    q.put(("done", None, None))


# ── GPU Inference ──────────────────────────────────────────────────────

def get_batch_embeddings(model, batched_pixels: torch.Tensor, device: str,
                         is_adapted: bool = False) -> np.ndarray:
    """Get V-JEPA 2 embeddings for a batch of processed videos.

    Uses the model's own parameter dtype (fp16 for V-JEPA 2.0 HF, bf16 for V-JEPA 2.1
    native — see #44) for both the input cast and autocast context. Hardcoded fp16 would
    crash `conv3d` in patch_embed when model is bf16 (input/weight dtype mismatch).
    """
    # Detect model dtype — handles torch.compile-wrapped models via `_orig_mod` if present
    m = getattr(model, "_orig_mod", model)
    model_dtype = next(m.parameters()).dtype
    pixel_values = batched_pixels.to(device=device, dtype=model_dtype)
    with torch.no_grad():
        if is_adapted:
            # Native vjepa2 VisionTransformer: forward(x) → (B, N, D) tensor
            if pixel_values.ndim == 5 and pixel_values.shape[2] in (1, 3):
                pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
            outputs = model(pixel_values)
            embeddings = outputs.mean(dim=1).float().cpu().numpy()
        else:
            # HF AutoModel: forward(pixel_values_videos=...) → ModelOutput
            with torch.amp.autocast("cuda", dtype=model_dtype):
                outputs = model(pixel_values_videos=pixel_values, skip_predictor=True)
                embeddings = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
    return embeddings


# ═════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR / WORKER (subprocess pattern for HF stream resilience)
# ═════════════════════════════════════════════════════════════════════════

def _resolve_model(user_model) -> tuple:
    """Normalize args.model → (model_path or None, is_adapted bool).

    - user_model is None: no --model passed → load frozen native from YAML checkpoint_path.
      is_adapted=False; model_path=None. Worker's `elif hf_model_id is None` branch handles it.
    - user_model is a .pt path that exists: adapted student from m09 training.
      is_adapted=True; model_path=Path(...). Worker's `if is_adapted` branch handles it.
    - Any other string (HF model id): frozen HF AutoModel path.
      is_adapted=False; model_path=Path(user_model) for display only.
    """
    if user_model is None:
        return None, False
    p = Path(user_model)
    return p, (p.suffix == ".pt" and p.exists())


def orchestrator_main(args):
    """Spawn worker subprocesses every ENGINE_RESTART_EVERY clips.

    Each worker gets fresh HF connections + GPU state. On stream stall
    (10-min producer timeout), worker exits, orchestrator respawns from checkpoint.
    """
    output_dir = get_module_output_dir("m05_vjepa_embed", args.subset, sanity=args.SANITY, poc=args.POC)

    # V-JEPA 2.1 has `hf_model_id: null` in its YAML → VJEPA_MODEL_ID is None → args.model is None.
    # In that case we're loading a frozen native checkpoint (from YAML checkpoint_path in the
    # worker's native-frozen branch, NOT the adapted-student branch). See `_resolve_model`.
    model_path, is_adapted = _resolve_model(args.model)
    encoder_name = getattr(args, 'encoder', None) or ("vjepa_adapted" if is_adapted else "vjepa")
    from utils.config import get_encoder_info
    embed_suffix = get_encoder_info(encoder_name)["suffix"]
    ckpt_fp = _checkpoint_fingerprint(model_path, is_adapted)
    embeddings_file = output_dir / f"embeddings{embed_suffix}.npy"
    checkpoint_file = output_dir / f".m05_checkpoint{embed_suffix}{ckpt_fp}.npz"

    print(f"Output: {embeddings_file}")
    if args.subset:
        print(f"[POC] Subset: {args.subset}")

    # iter11 #80 provenance gate — prevent variant-boundary .npy collision.
    # Bug: in paired_eval_10k v12, after v14 completed m05 and wrote
    #   outputs/full/m05_vjepa_embed/embeddings_vjepa_2_1_surgical.npy,
    # v15a's m05 saw the same-named .npy and verify_or_skip returned True →
    # v15a silently reused v14's embeddings → paired-bootstrap numbers
    # identical to v14 across mAP/Cycle/nDCG. Root cause: the output .npy
    # filename carries no per-variant fingerprint (unlike the checkpoint).
    # Fix (adapted-model only): if .npy exists but the CURRENT variant's
    # fingerprinted ckpt `.m05_checkpoint_*_<fp>.npz` is absent, the .npy
    # was produced by a DIFFERENT variant → bypass verify_or_skip so the
    # fresh np.save below atomically overwrites the stale .npy. Frozen
    # models (is_adapted=False) have no fingerprint, so the check is
    # skipped and behavior is unchanged for them.
    stale_npy = False
    if is_adapted and embeddings_file.exists() and not checkpoint_file.exists():
        print(f"[m05 provenance] .npy exists but ckpt for current fp={ckpt_fp} "
              f"missing → .npy belongs to a different variant. Forcing recompute "
              f"(fresh np.save will overwrite atomically; no rm in .sh needed).")
        stale_npy = True

    # Output-exists guard — bypassed when stale_npy is True.
    if not stale_npy:
        from utils.output_guard import verify_or_skip
        if verify_or_skip(output_dir, {
            "embeddings": embeddings_file,
            "paths": embeddings_file.with_suffix('.paths.npy'),
        }, label="m05 embed"):
            return

    # Determine total clips
    subset_keys = load_subset(args.subset) if args.subset else set()
    if args.SANITY:
        total_clips = get_sanity_clip_limit("embed")
    elif subset_keys:
        total_clips = len(subset_keys)
    else:
        total_clips = get_total_clips(local_data=getattr(args, 'local_data', None))
        if total_clips == 0:
            print("FATAL: Cannot determine clip count. Use --subset or --local-data with manifest.json")
            sys.exit(1)

    print(f"Clip limit: {total_clips:,}")
    print(f"Streaming from: {HF_DATASET_REPO}")

    # Load checkpoint to determine progress
    _, _, skip_count = load_checkpoint(checkpoint_file)
    if skip_count >= total_clips:
        print(f"All clips processed ({skip_count:,}/{total_clips:,}). Running post-processing...")
    else:
        if skip_count > 0:
            print(f"Resuming from checkpoint: {skip_count:,}/{total_clips:,} clips")

        segment_idx = 0
        exit_reason = "complete"  # "complete" | "stuck_clips" | "worker_crash"
        while skip_count < total_clips:
            segment_size = min(ENGINE_RESTART_EVERY, total_clips - skip_count)
            segment_idx += 1
            print(f"\n{'='*60}")
            print(f"WORKER {segment_idx}: embeddings {skip_count:,} → {skip_count + segment_size:,}")
            print(f"{'='*60}")

            cmd = [
                sys.executable, "-u", os.path.abspath(__file__),
                "--_worker",
                "--start-from", str(skip_count),
                "--process-count", str(segment_size),
            ]
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(args.batch_size)])
            if args.SANITY:
                cmd.append("--SANITY")
            if args.POC:
                cmd.append("--POC")
            if args.FULL:
                cmd.append("--FULL")
            if args.subset:
                cmd.extend(["--subset", args.subset])
            if args.no_wandb:
                cmd.append("--no-wandb")
            if args.gpu_mem is not None:
                cmd.extend(["--gpu-mem", str(args.gpu_mem)])
            if args.model != VJEPA_MODEL_ID:
                cmd.extend(["--model", args.model])
            if getattr(args, 'local_data', None):
                cmd.extend(["--local-data", args.local_data])
            if getattr(args, 'encoder', None):
                cmd.extend(["--encoder", args.encoder])
            # Propagate cache policy to worker (iter11 delete-protection gate).
            if getattr(args, 'cache_policy', None):
                cmd.extend(["--cache-policy", args.cache_policy])
            if getattr(args, 'shuffle', False):
                cmd.append("--shuffle")

            result = subprocess.run(cmd)

            _, _, new_count = load_checkpoint(checkpoint_file)
            if new_count > skip_count:
                skip_count = new_count
                print(f"Worker done. Progress: {skip_count:,}/{total_clips:,}")
            elif result.returncode != 0:
                print(f"Worker failed (exit {result.returncode}). Resume with same command.")
                exit_reason = "worker_crash"
                break
            else:
                # Worker exited cleanly but made no progress — the remaining
                # (total_clips - skip_count) clips all failed to decode/embed
                # (corrupt MP4, unsupported codec, etc.). Advance past the stuck
                # segment and record the reason so post-processing can decide
                # partial-success vs FATAL.
                print(f"Worker exited cleanly with no progress: "
                      f"{total_clips - skip_count:,} remaining clips appear irrecoverable.")
                skip_count = total_clips
                exit_reason = "stuck_clips"

    # ── Post-processing: dedupe + save final output ──
    all_embeddings, all_keys, final_count = load_checkpoint(checkpoint_file)
    if not all_embeddings:
        print("ERROR: No embeddings collected.")
        sys.exit(1)

    # GUARD: distinguish worker-crash (partial data is suspect) from stuck-clips
    # (partial data is complete; some inputs just can't be decoded). Two-tier:
    #   - final < 80% (emergency floor): FATAL regardless — too little signal.
    #   - 80%-95% + exit_reason=="worker_crash": FATAL (don't save crash-truncated).
    #   - 80%-95% + exit_reason=="stuck_clips": PARTIAL OK — save .npy + write
    #     failed_clip_keys.json so downstream paired eval aligns on embedded keys only.
    EMERGENCY_FLOOR = 0.80
    NORMAL_FLOOR = 0.95
    pct = final_count / total_clips
    if pct < EMERGENCY_FLOOR:
        print(f"FATAL: Only {final_count:,}/{total_clips:,} ({pct*100:.0f}%) < emergency "
              f"floor {EMERGENCY_FLOOR*100:.0f}% (exit_reason={exit_reason}). Resume with same command.")
        sys.exit(1)
    if pct < NORMAL_FLOOR and exit_reason == "worker_crash":
        print(f"FATAL: {final_count:,}/{total_clips:,} ({pct*100:.0f}%) + worker_crash — "
              f"partial data likely truncated by SIGKILL/SIGSEGV; not safe to promote to .npy. "
              f"Resume with same command.")
        sys.exit(1)
    if pct < NORMAL_FLOOR:
        # Partial success (stuck_clips path). Save .npy + failed_clip_keys.json.
        n_failed = total_clips - final_count
        print(f"⚠️  PARTIAL SUCCESS: {final_count:,}/{total_clips:,} ({pct*100:.1f}%) embedded; "
              f"{n_failed:,} clips irrecoverably failed decode (exit_reason=stuck_clips). "
              f"Proceeding with partial .npy — downstream (paired BCa, FAISS) aligns on "
              f"embedded keys only.")
        if args.subset:
            try:
                subset_keys = load_subset(args.subset)
                embedded = set(all_keys)
                failed = [k for k in subset_keys if k not in embedded]
                failed_path = output_dir / f"failed_clip_keys_{args.encoder}.json"
                with open(failed_path, 'w') as f:
                    json.dump({"n_failed": len(failed), "n_embedded": final_count,
                               "n_total": total_clips, "failed_clip_keys": failed}, f, indent=2)
                print(f"  Failed-clip manifest: {failed_path}")
            except Exception as e:
                print(f"  WARN: could not write failed_clip_keys.json: {e}")

    embeddings = np.stack(all_embeddings).astype(np.float32)
    clip_keys = all_keys

    print("\n=== Processing Stats ===")
    print(f"Total clips:     {len(clip_keys):,}")
    print(f"Embedding shape: {embeddings.shape}")

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, embeddings)
    np.save(embeddings_file.with_suffix('.paths.npy'), np.array(clip_keys, dtype=object))

    # iter11 META-fix: the v1→v10 paired_eval cycle lost ~10 h because this unlink
    # destroyed the checkpoint while the script wiped the .npy elsewhere — no durable
    # state survived a round-trip. Now gated: unlinks only if user typed `2` at the
    # .sh prompt (args.cache_policy=2/recompute). Default (1/keep) preserves checkpoint
    # as a second durable backup alongside the .npy.
    from utils.cache_policy import guarded_delete
    guarded_delete(checkpoint_file, args.cache_policy,
                   label=f"m05 checkpoint ({args.encoder})")

    print("\n=== EMBEDDING COMPLETE ===")
    print(f"Saved: {embeddings_file}")
    print(f"Shape: {embeddings.shape}")
    print(f"Unique clips: {len(clip_keys)}")

    wb_run = init_wandb("m05", "COMPLETE", config=vars(args),
                        enabled=not args.no_wandb)
    log_metrics(wb_run, {
        "total_clips": len(clip_keys),
        "embedding_dim": embeddings.shape[1],
    })
    log_artifact(wb_run, "embeddings", str(embeddings_file))
    log_artifact(wb_run, "paths", str(embeddings_file.with_suffix('.paths.npy')))
    finish_wandb(wb_run)


def worker_main(args):
    """Worker subprocess: load V-JEPA, process segment, save checkpoint, exit."""
    cleanup_temp()
    check_gpu()
    device = "cuda"

    if args.batch_size is None:
        # Read from: 1) profiler JSON, 2) pipeline.yaml, 3) linear estimate
        profile_path = Path("outputs/profile/inference/vjepa2/profile_data.json")
        if profile_path.exists():
            _profile = json.load(open(profile_path))
            if "optimal_bs" in _profile:
                args.batch_size = _profile["optimal_bs"]
                print(f"Batch size: {args.batch_size} (from inference/vjepa2 profiler)")
        if args.batch_size is None:
            # iter10 #78 fix: fail-loud on missing pipeline.yaml keys per CLAUDE.md §5.
            # Prior code used `.get("gpu", {}).get("inference_vjepa_bs")` which returned
            # None on schema drift → silently fell through to linear estimate → wrong BS,
            # potential OOM or throughput loss, no warning. Now yaml is mandatory.
            args.batch_size = get_pipeline_config()["gpu"]["inference_vjepa_bs"]
            print(f"Batch size: {args.batch_size} (from pipeline.yaml)")

    # Adapted models eval at 64f (VJEPA_FRAMES_PER_CLIP) but frozen HF models also use 64f.
    # The difference: adapted uses native vjepa2 (sdp_kernel patched), frozen uses HF AutoModel.
    # They have different VRAM profiles → separate BS configs in pipeline.yaml.
    model_path, is_adapted_pre = _resolve_model(args.model)
    # V-JEPA 2.1 native-frozen (args.model=None AND YAML hf_model_id=null) uses the SAME
    # native forward path + 64-frame inference as adapted students → same VRAM profile,
    # so apply the adapted BS cap here too (#46). Without this, args.batch_size stays at
    # the 96GB-profiler value (e.g. 176) and OOMs on 24GB at BS=100 on 2B bf16.
    _mcfg_early = get_model_config(getattr(args, "model_config", None))["model"]
    uses_native_fwd = is_adapted_pre or (args.model is None and _mcfg_early["hf_model_id"] is None)
    _gpu_cfg = get_pipeline_config()["gpu"]
    sizer = None  # Only used on native-fwd path (HF AutoModel handles its own mem via autocast)
    if uses_native_fwd:
        # Producer fills batches at the MAX cap; consumer sub-batches via AdaptiveBatchSizer
        # starting at the yaml-configured initial size, growing until gpu_memory_target × VRAM.
        # All three values come from configs/pipeline.yaml — no hardcoded Python defaults (#46).
        adapted_bs = _gpu_cfg["inference_adapted_bs"]
        initial_bs = _gpu_cfg["inference_vjepa_initial_bs"]
        mem_target = _gpu_cfg["gpu_memory_target"]
        if adapted_bs < args.batch_size:
            why = "adapted model" if is_adapted_pre else "V-JEPA 2.1 native (same native fwd as adapted)"
            print(f"Batch size: {args.batch_size} → {adapted_bs} ({why}, from pipeline.yaml)")
            args.batch_size = adapted_bs
        sizer = AdaptiveBatchSizer(initial_size=initial_bs, min_size=1,
                                   max_size=adapted_bs, memory_cap=mem_target)
        print(f"AdaptiveBatchSizer: start={initial_bs}, max={adapted_bs}, "
              f"target VRAM={mem_target:.0%} (from pipeline.yaml)")

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb_run = init_wandb("m05", mode,
                        config={"start_from": args.start_from,
                                "process_count": args.process_count},
                        enabled=not args.no_wandb)

    output_dir = get_module_output_dir("m05_vjepa_embed", args.subset, sanity=args.SANITY, poc=args.POC)
    # Match orchestrator's encoder-aware checkpoint path
    model_path, is_adapted = _resolve_model(args.model)
    encoder_name = getattr(args, 'encoder', None) or ("vjepa_adapted" if is_adapted else "vjepa")
    from utils.config import get_encoder_info
    embed_suffix = get_encoder_info(encoder_name)["suffix"]
    ckpt_fp = _checkpoint_fingerprint(model_path, is_adapted)
    checkpoint_file = output_dir / f".m05_checkpoint{embed_suffix}{ckpt_fp}.npz"
    subset_keys = load_subset(args.subset) if args.subset else set()

    # Load checkpoint for resume (processed_keys used by producer to skip done clips)
    all_embeddings, all_keys, resume_count = load_checkpoint(checkpoint_file)
    processed_keys = set(all_keys)

    clip_limit = args.process_count
    if clip_limit <= 0:
        print("No clips to process.")
        finish_wandb(wb_run)
        return

    if args.SANITY:
        args.batch_size = min(args.batch_size, 2)

    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print("FATAL: flash-attn not installed.")
        print("V-JEPA 2 ViT-G requires Flash Attention 2 for memory-efficient inference.")
        print("")
        print("Install via setup_env_uv.sh --gpu (downloads pre-built wheel), or manually:")
        print("  pip install flash-attn --no-build-isolation")
        sys.exit(1)

    try:
        model_path, is_adapted = _resolve_model(args.model)

        # Load model config to determine architecture
        mcfg = get_model_config(getattr(args, "model_config", None))["model"]
        arch = mcfg["arch"]
        hf_model_id = mcfg["hf_model_id"]
        crop_size = mcfg["crop_size"]

        if is_adapted:
            # Adapted encoder: use vjepa2's native VisionTransformer (same keys as m09 training)
            # HF AutoModel has different key format (split QKV, renamed layers) — incompatible
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "student_state_dict" in ckpt:
                state_dict = ckpt["student_state_dict"]
            else:
                state_dict = ckpt

            print(f"Adapted model ({arch}): eval at {VJEPA_FRAMES_PER_CLIP} frames (matching frozen baseline)")

            from utils.vjepa2_imports import get_vit_by_arch
            vit_constructor = get_vit_by_arch(arch)

            model = vit_constructor(
                img_size=(crop_size, crop_size), patch_size=16, num_frames=VJEPA_FRAMES_PER_CLIP,
                tubelet_size=2, use_sdpa=True, use_silu=False, wide_silu=True,
                uniform_power=False, use_rope=True,
            )
            msg = model.load_state_dict(state_dict, strict=False)
            loaded = len(state_dict) - len(msg.unexpected_keys)
            total = len(list(model.state_dict().keys()))
            print(f"Adapted encoder: loaded {loaded}/{total} params "
                  f"(missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
            if loaded < total * 0.9:
                print(f"FATAL: Only {loaded}/{total} params loaded — key mismatch!")
                print(f"  Checkpoint keys sample: {list(state_dict.keys())[:3]}")
                print(f"  Model keys sample:      {list(model.state_dict().keys())[:3]}")
                sys.exit(1)

            model = model.to(device=device, dtype=torch.float16)
            # V-JEPA 2.1 has no HF release → use 2.0's processor (same resolution, same normalization)
            proc_id = hf_model_id if hf_model_id else "facebook/vjepa2-vitg-fpc64-384"
            processor = AutoVideoProcessor.from_pretrained(proc_id)

        elif hf_model_id is None:
            # Native frozen model (V-JEPA 2.1 — no HuggingFace release, load from checkpoint)
            ckpt_path = Path(mcfg["checkpoint_path"])
            if not ckpt_path.is_absolute():
                ckpt_path = Path(__file__).parent.parent / ckpt_path
            if not ckpt_path.exists():
                print(f"FATAL: V-JEPA 2.1 checkpoint not found: {ckpt_path}")
                print(f"  Download: wget {mcfg['checkpoint_url']} -P checkpoints/")
                sys.exit(1)

            print(f"Frozen model ({arch}): native checkpoint at {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # Prefer target_encoder (EMA teacher — best quality), fallback to encoder
            state_dict = ckpt.get("target_encoder", ckpt.get("encoder", ckpt))
            # Strip DDP/wrapper prefixes
            state_dict = {k.replace("module.", "").replace("backbone.", ""): v
                          for k, v in state_dict.items()}

            from utils.vjepa2_imports import get_vit_by_arch
            vit_constructor = get_vit_by_arch(arch)
            model = vit_constructor(
                img_size=(crop_size, crop_size), patch_size=16, num_frames=VJEPA_FRAMES_PER_CLIP,
                tubelet_size=2, use_sdpa=True, use_silu=False, wide_silu=True,
                uniform_power=False, use_rope=True,
            )
            msg = model.load_state_dict(state_dict, strict=False)
            loaded = len(state_dict) - len(msg.unexpected_keys)
            total = len(list(model.state_dict().keys()))
            print(f"Frozen encoder: loaded {loaded}/{total} params "
                  f"(missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
            if loaded < total * 0.9:
                print(f"FATAL: Only {loaded}/{total} params loaded — key mismatch!")
                sys.exit(1)

            # bf16 for V-JEPA 2.1: same FA2 throughput as fp16 on Blackwell sm_120, wider dynamic
            # range avoids NaN on long video token sequences (#44). Paired with RoPE Q/K dtype
            # cast in deps/vjepa2 (`q, k = q.to(v.dtype), k.to(v.dtype)`) so SDPA sees consistent
            # dtypes and can dispatch to FA2 under torch.compile tracing.
            model = model.to(device=device, dtype=torch.bfloat16)
            model.eval()
            is_adapted = True  # Use the native forward path in get_batch_embeddings()
            processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-384")

        else:
            # Standard HF model (frozen baseline — V-JEPA 2.0 via HuggingFace)
            processor = AutoVideoProcessor.from_pretrained(hf_model_id)
            model = AutoModel.from_pretrained(
                hf_model_id,
                dtype=torch.float16,  # transformers 5.x: `torch_dtype` → `dtype` (#37)
                device_map="auto",
                attn_implementation="flash_attention_2",
            )

        model.eval()
        if is_adapted:
            # Monkey-patch deprecated sdp_kernel() to eliminate torch.compile graph breaks.
            # vjepa2 calls `with torch.backends.cuda.sdp_kernel():` (no args) which is a
            # no-op (enables all SDPA backends = default), but it creates graph breaks that
            # prevent inductor memory optimization (89GB instead of 30GB at BS=176).
            # Confirmed safe: all 7 call sites in deps/vjepa2 use bare sdp_kernel() (no args).
            # Reference: PyTorch Issue #130098.
            torch.backends.cuda.sdp_kernel = contextlib.nullcontext
        print("Applying torch.compile...")
        model = torch.compile(model)
        # Warmup: trigger compilation at small BS so inductor caches an efficient graph.
        # Matches profiler behavior (BS=4 warmup → 30GB at BS=176).
        print("Warmup: compiling inductor graph...")
        # Warmup tensor must match the model's own parameter dtype (fp16 for V-JEPA 2.0 HF,
        # bf16 for V-JEPA 2.1 native — see #44). Hardcoded fp16 would crash conv3d in
        # patch_embed with "Input type (Half) and bias type (BFloat16) should be the same".
        _m_un = getattr(model, "_orig_mod", model)
        _warmup_dtype = next(_m_un.parameters()).dtype
        warmup_t = torch.randn(2, 3, VJEPA_FRAMES_PER_CLIP, 384, 384,
                               dtype=_warmup_dtype, device=device)
        with torch.no_grad():
            if is_adapted:
                _ = model(warmup_t)
            else:
                _ = model(pixel_values_videos=warmup_t, skip_predictor=True)
        torch.cuda.synchronize()
        del warmup_t
        torch.cuda.empty_cache()
        print(f"Model loaded on {device} (dtype: {next(model.parameters()).dtype})")
    except Exception as e:
        print(f"FATAL: Model load failed: {e}")
        sys.exit(1)

    # Producer-consumer setup
    print("\n=== Streaming Config ===")
    print(f"batch_size:    {args.batch_size}")
    print(f"video_decoder: {'torchcodec (fast)' if _USE_TORCHCODEC else 'PyAV'}")
    print(f"prefetch:      {PREFETCH_QUEUE_SIZE} batches")
    print(f"checkpoint:    every {CHECKPOINT_EVERY} clips")

    tmp_base = output_dir / "tmp_m05"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)

    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_producer_thread,
        args=(processor, args.batch_size, tmp_dir, q, stop_event,
              clip_limit, subset_keys, VJEPA_FRAMES_PER_CLIP, processed_keys,
              getattr(args, 'local_data', None),
              getattr(args, 'shuffle', False)),
        daemon=True,
    )
    producer.start()

    start_time = time.time()
    last_window_time = start_time
    last_window_count = 0
    failed_count = 0
    checkpoint_thread = None
    pbar = tqdm(total=clip_limit, desc="m05_vjepa", unit="clip")

    try:
        while True:
            try:
                msg_type, batched_pixels, batch_keys = q.get(timeout=60)
            except queue.Empty:
                print("\nProducer timeout (10 min). Saving checkpoint...")
                break

            if msg_type == "done":
                break

            # Sub-batch the producer's batch using AdaptiveBatchSizer (native-fwd path only).
            # HF AutoModel path stays single-shot (its autocast handles memory internally).
            if sizer is None:
                embeddings = get_batch_embeddings(model, batched_pixels, device,
                                                  is_adapted=is_adapted)
            else:
                sub_embs = []
                i = 0
                while i < batched_pixels.shape[0]:
                    sub = batched_pixels[i : i + sizer.size]
                    oom = False
                    try:
                        sub_emb = get_batch_embeddings(model, sub, device, is_adapted=is_adapted)
                    except torch.cuda.OutOfMemoryError:
                        oom = True
                    # Cleanup OUTSIDE except (exception holds stack frame refs — see m04 pattern)
                    if oom:
                        del sub
                        gc.collect()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        if not sizer.on_oom():
                            raise torch.cuda.OutOfMemoryError(
                                f"AdaptiveBatchSizer at min_size={sizer.min_size} and still OOM")
                        continue  # retry same i with smaller sizer.size
                    sub_embs.append(sub_emb)
                    sizer.after_batch_success()
                    i += sub.shape[0]
                embeddings = np.concatenate(sub_embs, axis=0)

            for emb, key in zip(embeddings, batch_keys):
                all_embeddings.append(emb)
                all_keys.append(key)
                processed_keys.add(key)

            clips_this_run = len(all_embeddings) - resume_count
            pbar.update(len(batch_keys))

            # Windowed throughput
            now = time.time()
            window_elapsed = now - last_window_time
            window_clips = clips_this_run - last_window_count
            throughput = window_clips / window_elapsed if window_elapsed > 0 else 0
            pbar.set_postfix_str(f"{throughput:.1f} clips/s | failed={failed_count}")
            log_metrics(wb_run, {
                "clips_processed": len(all_embeddings),
                "throughput_clips_per_s": throughput,
                "failed": failed_count,
            })
            # Reset window every 30s
            if window_elapsed >= 30:
                last_window_time = now
                last_window_count = clips_this_run

            if len(all_embeddings) % CHECKPOINT_EVERY < args.batch_size:
                if checkpoint_thread and checkpoint_thread.is_alive():
                    checkpoint_thread.join()
                emb_snapshot = list(all_embeddings)
                keys_snapshot = list(all_keys)
                checkpoint_thread = threading.Thread(
                    target=save_checkpoint,
                    args=(emb_snapshot, keys_snapshot, checkpoint_file),
                    daemon=True,
                )
                checkpoint_thread.start()

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        stop_event.set()
    finally:
        pbar.close()
        stop_event.set()
        if checkpoint_thread and checkpoint_thread.is_alive():
            checkpoint_thread.join(timeout=30)
        save_checkpoint(all_embeddings, all_keys, checkpoint_file)
        producer.join(timeout=10)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            tmp_base.rmdir()
        except OSError as e:
            # Only swallow the benign "dir not empty" case — any other OSError
            # (permission, IO, device) must surface per CLAUDE.md §5 (no silent swallows).
            if e.errno != errno.ENOTEMPTY:
                raise
            print(f"[m05] tmp_base non-empty, leaving for next run: {tmp_base}")

    clips_this_run = len(all_embeddings) - resume_count
    if clips_this_run > 0:
        elapsed = time.time() - start_time
        print(f"\nSegment done: {clips_this_run:,} clips in {elapsed:.0f}s "
              f"({clips_this_run/elapsed:.2f} clips/s)")

    finish_wandb(wb_run)
    print("Worker exiting (GPU memory will be released).")

    # Force exit: torch.compile + CUDA atexit cleanup deadlocks on futex_wait_queue
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    # Enable expandable_segments BEFORE any CUDA allocation. Reduces OOM-retry failure
    # rate by letting the allocator grow existing segments instead of reserving new
    # large blocks that get fragmented. Matches PyTorch's own OOM error-message
    # recommendation. Idempotent — env var only read at first CUDA init.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cleanup_temp()
    parser = argparse.ArgumentParser(
        description="Generate V-JEPA 2 embeddings (GPU-only, HF WebDataset streaming)")
    parser.add_argument("--SANITY", action="store_true", help="Process 5 clips only")
    parser.add_argument("--POC", action="store_true", help="10K subset")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    parser.add_argument("--model", type=str, default=VJEPA_MODEL_ID,
                        help="Model ID or adapted checkpoint path")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (auto-computed if omitted)")
    parser.add_argument("--encoder", type=str, default=None,
                        help="Encoder name for output files (e.g., vjepa_lambda0)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle frame order (for temporal ablation)")
    add_model_config_arg(parser)
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    add_gpu_mem_arg(parser)

    # Internal worker args (spawned by orchestrator)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start-from", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--process-count", type=int, default=ENGINE_RESTART_EVERY,
                        help=argparse.SUPPRESS)
    # Cache-policy gate (iter11): every destructive delete in this module must route
    # through utils.cache_policy.guarded_delete(path, args.cache_policy, ...).
    # --cache-policy defaults to 1 (keep) so overnight re-runs never destroy cache.
    from utils.cache_policy import add_cache_policy_arg
    add_cache_policy_arg(parser)
    args = parser.parse_args()

    # Worker mode (spawned by orchestrator)
    if args._worker:
        worker_main(args)
        return

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    ensure_local_data(args)
    orchestrator_main(args)


if __name__ == "__main__":
    main()
