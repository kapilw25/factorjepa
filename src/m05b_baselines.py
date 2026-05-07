"""
Generate baseline embeddings: Random, Oracle, DINOv2, CLIP, Shuffled V-JEPA.
GPU-only for DINOv2/CLIP/Shuffled (no CPU fallback). Random + Oracle are CPU-safe.

USAGE (--profile-data optional; per-encoder profile JSON path; CLAUDE.md no-default rule):
    python -u src/m05b_baselines.py --encoder all --SANITY \
        --subset data/sanity_100_dense.json --local-data data/val_1k_local \
        --tags-json data/val_1k_local/tags.json \
        2>&1 | tee logs/m05b_all_sanity.log
    python -u src/m05b_baselines.py --encoder oracle --FULL \
        --subset data/ultra_hard_3066_eval.json --local-data data/ultra_hard_3066_local \
        --tags-json data/ultra_hard_3066_local/tags.json \
        2>&1 | tee logs/m05b_oracle_ultra_hard_3066.log
    python -u src/m05b_baselines.py --encoder all --FULL \
        --local-data data/full_local --tags-json data/full_local/tags.json \
        2>&1 | tee logs/m05b_all_full.log
"""
import argparse
import io
import json
import os
import queue
import shutil
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
    ENCODER_REGISTRY, VJEPA_EMBEDDING_DIM, VJEPA_FRAMES_PER_CLIP,
    check_gpu, load_subset, add_subset_arg, add_local_data_arg,
    get_module_output_dir, get_encoder_files,
    get_sanity_clip_limit, get_total_clips, get_pipeline_config,
    verify_npy_matches_subset,
)
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.gpu_batch import compute_batch_sizes, add_gpu_mem_arg, cuda_cleanup, cleanup_temp, AdaptiveBatchSizer
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, log_artifact, finish_wandb

# Shared video I/O from utils (Rule 32: no cross-imports between m*.py)
from utils.video_io import get_clip_key, create_stream, decode_video_bytes
from utils.checkpoint import save_embedding_checkpoint as save_checkpoint, load_embedding_checkpoint as load_checkpoint

_pcfg_stream = get_pipeline_config()
DECODE_WORKERS = _pcfg_stream["streaming"]["decode_workers_embed"]
MAX_STREAM_RETRIES = _pcfg_stream["streaming"]["max_retries"]
CHECKPOINT_EVERY = _pcfg_stream["streaming"]["checkpoint_every"]
PREFETCH_QUEUE_SIZE = _pcfg_stream["streaming"]["prefetch_queue_embed"]

_create_stream = create_stream

# Lazy imports (GPU-only libs)
torch = None
_TORCH_IMPORTED = False


def _import_torch():
    global torch, _TORCH_IMPORTED
    if not _TORCH_IMPORTED:
        import torch as _torch
        torch = _torch
        _TORCH_IMPORTED = True
    return torch


# ── Random Baseline (CPU-safe) ───────────────────────────────────────

def generate_random(clip_keys: list, output_dir: Path, args):
    """Task 16: Random L2-normalized vectors. CPU-only, deterministic."""
    dim = VJEPA_EMBEDDING_DIM  # 1408 — match V-JEPA for apples-to-apples
    n = len(clip_keys)
    print(f"Generating {n:,} random embeddings (dim={dim}, seed=42)")

    rng = np.random.RandomState(42)
    embeddings = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings /= norms

    files = get_encoder_files("random", output_dir)
    np.save(files["embeddings"], embeddings)
    np.save(files["paths"], np.array(clip_keys, dtype=object))

    print(f"Saved: {files['embeddings']} ({embeddings.shape})")
    print(f"Saved: {files['paths']} ({len(clip_keys)} keys)")
    return embeddings, clip_keys


# ── Oracle Baseline (CPU-safe) ───────────────────────────────────────

def generate_oracle(clip_keys: list, output_dir: Path, args):
    """Oracle upper-bound: L2-normed multi-hot of tag fields.

    For each clip, encode `scene_type` (m06's primary metric `field`, line 319/352/383
    of m06_faiss_metrics.py) plus slice fields {weather, time_of_day, crowd_density,
    traffic_density, tour_type} as a concatenated one-hot vector; L2-normalize.
    By construction, FAISS kNN finds clips with identical tag-set → legacy
    retrieval scores approach 100% (modulo per-cluster size and tie-breaking).
    Establishes the theoretical retrieval ceiling on the legacy m06 metric.

    Aligns clip_keys → tags via Path(p).name == tags[i]["source_file"] (matches
    m06's filtering logic at m06_faiss_metrics.py:1163). CPU-only, deterministic.
    """
    tags_path = Path(args.tags_json)
    if not tags_path.exists():
        print(f"FATAL: --tags-json not found: {tags_path}")
        sys.exit(1)
    with open(tags_path) as f:
        tags = json.load(f)
    print(f"Loaded {len(tags):,} tags from {tags_path}")

    # Fail-loud upfront: every tag dict MUST have source_file (used for clip alignment).
    missing_src = [i for i, t in enumerate(tags) if "source_file" not in t]
    if missing_src:
        print(f"FATAL: {len(missing_src)} tag entries missing 'source_file' "
              f"(first 3 indices: {missing_src[:3]}). tags.json schema violation.")
        sys.exit(1)
    tag_by_source = {t["source_file"]: t for t in tags}

    # m06 uses scene_type primary; slice metrics use the rest. NO FALLBACK
    # (CLAUDE.md NO DEFAULT rule) — every tag dict MUST contain every field
    # with a non-None value. Fail-loud if any are missing.
    FIELDS = ["scene_type", "weather", "time_of_day", "crowd_density",
              "traffic_density", "tour_type"]
    for f in FIELDS:
        bad = [t.get("source_file", f"<idx={i}>")
               for i, t in enumerate(tags)
               if f not in t or t[f] is None]
        if bad:
            print(f"FATAL: field '{f}' missing/None in {len(bad)} tag entries "
                  f"(first 3: {bad[:3]}). Cannot build oracle multi-hot.")
            sys.exit(1)
    vocab = {f: sorted({str(t[f]) for t in tags}) for f in FIELDS}
    vocab_index = {f: {v: i for i, v in enumerate(vocab[f])} for f in FIELDS}
    field_offset = {}
    offset = 0
    for f in FIELDS:
        field_offset[f] = offset
        offset += len(vocab[f])
    dim = offset
    field_str = ", ".join(f"{f}:{len(vocab[f])}" for f in FIELDS)
    print(f"Oracle dim={dim} ({field_str})")

    # Fail-loud on clip-key alignment: every clip_key MUST resolve to a tag.
    # Skipping unmatched clips would silently emit zero-vectors → garbage kNN.
    n = len(clip_keys)
    embeddings = np.zeros((n, dim), dtype=np.float32)
    unmatched = []
    for i, p in enumerate(clip_keys):
        source = Path(p).name
        t = tag_by_source.get(source)
        if t is None:
            unmatched.append(p)
            continue
        for f in FIELDS:
            embeddings[i, field_offset[f] + vocab_index[f][str(t[f])]] = 1.0
    if unmatched:
        print(f"FATAL: {len(unmatched)}/{n} clip_keys have NO matching tag entry "
              f"(first 3: {unmatched[:3]}). Re-generate tags.json or fix subset.")
        sys.exit(1)
    print(f"Matched {n:,}/{n:,} clip_keys → tags via source_file")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings /= norms

    files = get_encoder_files("oracle", output_dir)
    files["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(files["embeddings"], embeddings)
    np.save(files["paths"], np.array(clip_keys, dtype=object))

    print(f"Saved: {files['embeddings']} ({embeddings.shape})")
    print(f"Saved: {files['paths']} ({len(clip_keys)} keys)")
    return embeddings, clip_keys


# ── Image Encoder Baselines (DINOv2, CLIP) ───────────────────────────

def _extract_middle_frame(video_tensor, num_frames: int):
    """Extract middle frame from (T, C, H, W) tensor -> PIL Image."""
    from PIL import Image

    mid_idx = video_tensor.shape[0] // 2
    frame = video_tensor[mid_idx]  # (C, H, W)
    if hasattr(frame, 'numpy'):
        frame = frame.numpy()
    # (C, H, W) -> (H, W, C)
    frame = np.transpose(frame, (1, 2, 0))
    return Image.fromarray(frame.astype(np.uint8))


def _decode_and_extract_frame(mp4_bytes: bytes, clip_key: str,
                               tmp_dir: str, num_frames: int):
    """Extract middle frame via decord (single-frame seek, no full decode).

    5-10x faster than PyAV full decode for image encoders (CLIP/DINOv2)
    that only need 1 frame per clip. Falls back to PyAV on decord failure.
    """
    from PIL import Image

    try:
        from decord import VideoReader
        vr = VideoReader(io.BytesIO(mp4_bytes))
        mid_idx = len(vr) // 2
        frame = vr[mid_idx].asnumpy()  # (H, W, C) uint8
        return Image.fromarray(frame), clip_key
    except Exception:
        # Fallback: full PyAV decode (slower but more robust)
        tensor = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames)
        if tensor is None:
            return None, clip_key
        return _extract_middle_frame(tensor, num_frames), clip_key


def _producer_image_baseline(processor, batch_size: int, tmp_dir: str,
                              q: queue.Queue, stop_event: threading.Event,
                              clip_limit: int, subset_keys: set,
                              processed_keys: set, num_frames: int,
                              local_data: str = None):
    """Parallel TAR reading + parallel decode for image encoders (CLIP/DINOv2).

    8 threads read TARs concurrently → shared clip queue → DECODE_WORKERS threads
    decode + extract middle frame → processor batches → GPU consumer queue.
    """
    produced = 0

    if local_data:
        # Parallel TAR reader: 8 threads reading different TARs simultaneously
        clip_q, tar_stop, reader = iter_clips_parallel(
            local_data, subset_keys=subset_keys, processed_keys=processed_keys)
    else:
        # Fallback: HF streaming (no parallel TAR reading)
        clip_q, tar_stop, _reader = None, None, None

    try:
        if clip_q is not None:
            # Fast path: parallel TAR reading → parallel decode → batch
            pending = []  # list of (mp4_bytes, clip_key)

            while produced < clip_limit and not stop_event.is_set():
                try:
                    item = clip_q.get(timeout=120)
                except queue.Empty:
                    print("  Parallel TAR reader timeout (2 min), flushing remaining")
                    break

                if item is None:  # sentinel: all TARs exhausted
                    break

                clip_key, mp4_bytes = item
                pending.append((mp4_bytes, clip_key))

                if len(pending) >= batch_size:
                    # Parallel decode + frame extraction
                    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                        futures = [
                            pool.submit(_decode_and_extract_frame, b, k, tmp_dir, num_frames)
                            for b, k in pending
                        ]
                        results = [f.result() for f in futures]

                    frames = [img for img, _ in results if img is not None]
                    keys = [k for img, k in results if img is not None]

                    if frames:
                        inputs = processor(images=frames, return_tensors="pt")
                        q.put(("batch", inputs, keys[:]))
                        produced += len(frames)

                    pending = []

                    if produced >= clip_limit:
                        break

            # Final partial batch
            if pending and not stop_event.is_set():
                with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                    futures = [
                        pool.submit(_decode_and_extract_frame, b, k, tmp_dir, num_frames)
                        for b, k in pending
                    ]
                    results = [f.result() for f in futures]

                frames = [img for img, _ in results if img is not None]
                keys = [k for img, k in results if img is not None]

                if frames:
                    inputs = processor(images=frames, return_tensors="pt")
                    q.put(("batch", inputs, keys[:]))
                    produced += len(frames)

        else:
            # HF streaming fallback (no local data)
            retries = 0
            while produced < clip_limit and not stop_event.is_set():
                try:
                    ds = _create_stream(0, local_data=local_data)
                    pending_bytes = []
                    pending_keys = []

                    for example in ds:
                        if stop_event.is_set():
                            break

                        clip_key = get_clip_key(example)
                        if subset_keys and clip_key not in subset_keys:
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
                            with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                                futures = [
                                    pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
                                    for b, k in zip(pending_bytes, pending_keys)
                                ]
                                results = [(f.result(), k) for f, k in zip(futures, pending_keys)]

                            frames = [_extract_middle_frame(t, num_frames) for t, k in results if t is not None]
                            keys = [k for t, k in results if t is not None]

                            if frames:
                                inputs = processor(images=frames, return_tensors="pt")
                                q.put(("batch", inputs, keys[:]))
                                produced += len(frames)

                            pending_bytes, pending_keys = [], []
                            if produced >= clip_limit:
                                break

                    break  # stream exhausted

                except (ConnectionError, TimeoutError, OSError) as e:
                    retries += 1
                    if retries > MAX_STREAM_RETRIES:
                        print(f"  ERROR: stream failed after {MAX_STREAM_RETRIES} retries: {e}")
                        break
                    wait = min(2 ** retries, 60)
                    print(f"  WARN: stream error ({e}), retry {retries}/{MAX_STREAM_RETRIES}")
                    time.sleep(wait)

    except Exception as e:
        print(f"  ERROR: unexpected producer error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tar_stop:
            tar_stop.set()

    q.put(("done", None, None))


def generate_dinov2(output_dir: Path, args, clip_limit: int, subset_keys: set):
    """DINOv2 ViT-g/14 — middle frame, CLS token, 1536-dim."""
    torch = _import_torch()
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda"
    model_id = ENCODER_REGISTRY["dinov2"]["model_id"]
    files = get_encoder_files("dinov2", output_dir)
    checkpoint_file = output_dir / ".m05b_dinov2_checkpoint.npz"

    print(f"Loading model: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float16,  # #37
                                      device_map="auto",
                                      attn_implementation="flash_attention_2")
    model.eval()
    print("Applying torch.compile (first batch will be slow due to compilation)...")
    model = torch.compile(model)
    dim = ENCODER_REGISTRY["dinov2"]["dim"]
    print(f"DINOv2 loaded (dim={dim}, dtype=float16, FA2+compiled)")

    all_embeddings, all_keys, resume_count = load_checkpoint(checkpoint_file)
    processed_keys = set(all_keys)

    # Read from: 1) CLI arg, 2) --profile-data CLI arg, 3) pipeline.yaml, 4) linear estimate
    batch_size = None
    if args.batch_size:
        batch_size = args.batch_size
    else:
        if args.profile_data:
            _prof_path = Path(args.profile_data)
            if _prof_path.exists():
                _prof = json.load(open(_prof_path))
                if "optimal_bs" in _prof:
                    batch_size = _prof["optimal_bs"]
                    print(f"Batch size: {batch_size} (from --profile-data {_prof_path})")
        if batch_size is None:
            batch_size = get_pipeline_config()["gpu"].get("inference_dinov2_bs",
                         compute_batch_sizes(gpu_vram_gb=args.gpu_mem)["image_encoder"])
    if args.SANITY:
        batch_size = min(batch_size, 4)

    # Adaptive sub-batcher: initial from yaml, max = producer BS (hard ceiling),
    # VRAM ceiling from universal `gpu_memory_target` (#47). Same pattern as m05/m04.
    _gpu_cfg = get_pipeline_config()["gpu"]
    sizer = AdaptiveBatchSizer(
        initial_size=min(_gpu_cfg["inference_dinov2_initial_bs"], batch_size),
        min_size=1, max_size=batch_size,
        memory_cap=_gpu_cfg["gpu_memory_target"])
    print(f"AdaptiveBatchSizer: start={sizer.size}, max={batch_size}, "
          f"target VRAM={_gpu_cfg['gpu_memory_target']:.0%} (from pipeline.yaml)")

    tmp_base = output_dir / "tmp_m05b"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)
    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_producer_image_baseline,
        args=(processor, batch_size, tmp_dir, q, stop_event, clip_limit,
              subset_keys, processed_keys, VJEPA_FRAMES_PER_CLIP,
              getattr(args, 'local_data', None)),
        daemon=True,
    )
    producer.start()

    start_time = time.time()
    last_window_count = len(all_embeddings)
    last_window_time = start_time
    pbar = tqdm(total=clip_limit, initial=resume_count,
                desc="m05b dinov2", unit="clip")
    try:
        while True:
            try:
                msg_type, batch_inputs, batch_keys = q.get(timeout=60)
            except queue.Empty:
                print("\nProducer timeout (10 min). Saving checkpoint...")
                break

            if msg_type == "done":
                break

            inputs = {k: v.to(device) for k, v in batch_inputs.items() if hasattr(v, 'to')}
            # Sub-batch via AdaptiveBatchSizer — first stacked tensor key drives slicing.
            _tensor_keys = [k for k, v in inputs.items() if hasattr(v, 'shape')]
            _total = inputs[_tensor_keys[0]].shape[0] if _tensor_keys else len(batch_keys)
            _sub_embs = []
            _i = 0
            while _i < _total:
                _sub = {k: (v[_i:_i + sizer.size] if k in _tensor_keys else v)
                        for k, v in inputs.items()}
                _oom = False
                try:
                    with torch.no_grad():
                        _out = model(**_sub)
                        _emb_sub = _out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
                except torch.cuda.OutOfMemoryError:
                    _oom = True
                if _oom:
                    cuda_cleanup()
                    if not sizer.on_oom():
                        raise torch.cuda.OutOfMemoryError("m05b DINOv2 at min sub-batch")
                    continue
                _sub_embs.append(_emb_sub)
                sizer.after_batch_success()
                _i += _sub[_tensor_keys[0]].shape[0] if _tensor_keys else 1
            # iter13 v13 FIX-14 (2026-05-07): FAIL HARD on zero embeddings.
            # Previous `np.empty((0, _out.last_hidden_state.shape[-1]))` fallback
            # silently saved a zero-row .npy, which downstream m06/m07 hit as
            # cryptic shape errors. Plus _out may be undefined if every forward
            # pass OOM'd before assignment → NameError. Either way, garbage in.
            if not _sub_embs:
                raise RuntimeError(
                    "m05b DINOv2 batch produced 0 embeddings — sub-batch loop "
                    "never accumulated. Likely cause: empty input batch OR every "
                    "sub-batch OOM'd at min size (sizer.on_oom raises before this)."
                )
            emb = np.concatenate(_sub_embs, axis=0)

            # L2-normalize
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb /= norms

            # Enforce clip limit: truncate batch if it would overshoot
            remaining = clip_limit - len(all_embeddings)
            for e, k in zip(emb[:remaining], batch_keys[:remaining]):
                all_embeddings.append(e)
                all_keys.append(k)
                processed_keys.add(k)

            pbar.update(min(len(batch_keys), remaining))

            if len(all_embeddings) >= clip_limit:
                stop_event.set()
                break

            now = time.time()
            window_clips = len(all_embeddings) - last_window_count
            window_time = now - last_window_time
            throughput = window_clips / window_time if window_time > 0 else 0
            last_window_count = len(all_embeddings)
            last_window_time = now
            pbar.set_postfix({"rate": f"{throughput:.1f}/s"})

            if len(all_embeddings) % CHECKPOINT_EVERY < batch_size:
                save_checkpoint(list(all_embeddings), list(all_keys), checkpoint_file)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        stop_event.set()
    finally:
        pbar.close()
        stop_event.set()
        save_checkpoint(all_embeddings, all_keys, checkpoint_file)
        producer.join(timeout=10)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _finalize(all_embeddings, all_keys, files, checkpoint_file,
              cache_policy=args.cache_policy,
              subset_path=args.subset, label="m05b dinov2")
    return np.stack(all_embeddings).astype(np.float32), all_keys


def generate_clip(output_dir: Path, args, clip_limit: int, subset_keys: set):
    """Task 19: CLIP ViT-L/14 — middle frame, image features, 768-dim."""
    torch = _import_torch()
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda"
    model_id = ENCODER_REGISTRY["clip"]["model_id"]
    files = get_encoder_files("clip", output_dir)
    checkpoint_file = output_dir / ".m05b_clip_checkpoint.npz"

    print(f"Loading model: {model_id}")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, dtype=torch.float16,  # #37
                                      device_map="auto",
                                      attn_implementation="sdpa")
    model.eval()
    print("Applying torch.compile (first batch will be slow due to compilation)...")
    model = torch.compile(model)
    print("CLIP loaded (dim=768, dtype=float16, SDPA+compiled)")

    all_embeddings, all_keys, resume_count = load_checkpoint(checkpoint_file)
    processed_keys = set(all_keys)

    batch_size = args.batch_size or compute_batch_sizes(gpu_vram_gb=args.gpu_mem)["image_encoder"]
    if args.SANITY:
        batch_size = min(batch_size, 4)

    # AdaptiveBatchSizer (#47) — same universal pattern as DINOv2 runner above.
    _gpu_cfg = get_pipeline_config()["gpu"]
    sizer = AdaptiveBatchSizer(
        initial_size=min(_gpu_cfg["inference_clip_initial_bs"], batch_size),
        min_size=1, max_size=batch_size,
        memory_cap=_gpu_cfg["gpu_memory_target"])
    print(f"AdaptiveBatchSizer: start={sizer.size}, max={batch_size}, "
          f"target VRAM={_gpu_cfg['gpu_memory_target']:.0%} (from pipeline.yaml)")

    tmp_base = output_dir / "tmp_m05b"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)
    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_producer_image_baseline,
        args=(processor, batch_size, tmp_dir, q, stop_event, clip_limit,
              subset_keys, processed_keys, VJEPA_FRAMES_PER_CLIP,
              getattr(args, 'local_data', None)),
        daemon=True,
    )
    producer.start()

    start_time = time.time()
    last_window_count = len(all_embeddings)
    last_window_time = start_time
    pbar = tqdm(total=clip_limit, initial=resume_count,
                desc="m05b clip", unit="clip")
    try:
        while True:
            try:
                msg_type, batch_inputs, batch_keys = q.get(timeout=60)
            except queue.Empty:
                print("\nProducer timeout (10 min). Saving checkpoint...")
                break

            if msg_type == "done":
                break

            inputs = {k: v.to(device) for k, v in batch_inputs.items() if hasattr(v, 'to')}
            # Sub-batch via sizer (#47) — same pattern as DINOv2 path.
            _tensor_keys = [k for k, v in inputs.items() if hasattr(v, 'shape')]
            _total = inputs[_tensor_keys[0]].shape[0] if _tensor_keys else len(batch_keys)
            _sub_embs = []
            _i = 0
            while _i < _total:
                _sub = {k: (v[_i:_i + sizer.size] if k in _tensor_keys else v)
                        for k, v in inputs.items()}
                _oom = False
                try:
                    with torch.no_grad():
                        _emb_sub = model.get_image_features(**_sub).cpu().numpy().astype(np.float32)
                except torch.cuda.OutOfMemoryError:
                    _oom = True
                if _oom:
                    cuda_cleanup()
                    if not sizer.on_oom():
                        raise torch.cuda.OutOfMemoryError("m05b CLIP at min sub-batch")
                    continue
                _sub_embs.append(_emb_sub)
                sizer.after_batch_success()
                _i += _sub[_tensor_keys[0]].shape[0] if _tensor_keys else 1
            # iter13 v13 FIX-14 (2026-05-07): mirror DINOv2 path — FAIL HARD on
            # zero embeddings instead of silently writing a zero-row .npy.
            if not _sub_embs:
                raise RuntimeError(
                    "m05b CLIP batch produced 0 embeddings — sub-batch loop "
                    "never accumulated. Likely cause: empty input batch OR every "
                    "sub-batch OOM'd at min size (sizer.on_oom raises before this)."
                )
            emb = np.concatenate(_sub_embs, axis=0)

            # L2-normalize
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb /= norms

            # Enforce clip limit: truncate batch if it would overshoot
            remaining = clip_limit - len(all_embeddings)
            for e, k in zip(emb[:remaining], batch_keys[:remaining]):
                all_embeddings.append(e)
                all_keys.append(k)
                processed_keys.add(k)

            pbar.update(min(len(batch_keys), remaining))

            if len(all_embeddings) >= clip_limit:
                stop_event.set()
                break

            now = time.time()
            window_clips = len(all_embeddings) - last_window_count
            window_time = now - last_window_time
            throughput = window_clips / window_time if window_time > 0 else 0
            last_window_count = len(all_embeddings)
            last_window_time = now
            pbar.set_postfix({"rate": f"{throughput:.1f}/s"})

            if len(all_embeddings) % CHECKPOINT_EVERY < batch_size:
                save_checkpoint(list(all_embeddings), list(all_keys), checkpoint_file)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        stop_event.set()
    finally:
        pbar.close()
        stop_event.set()
        save_checkpoint(all_embeddings, all_keys, checkpoint_file)
        producer.join(timeout=10)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _finalize(all_embeddings, all_keys, files, checkpoint_file,
              cache_policy=args.cache_policy,
              subset_path=args.subset, label="m05b clip")
    return np.stack(all_embeddings).astype(np.float32), all_keys


# ── Shuffled V-JEPA ──────────────────────────────────────────────────

def _producer_shuffled_vjepa(processor, batch_size: int, tmp_dir: str,
                              q: queue.Queue, stop_event: threading.Event,
                              clip_limit: int, subset_keys: set,
                              processed_keys: set, num_frames: int,
                              local_data: str = None):
    """Stream from HF (or local shards), decode, shuffle frame order, process for V-JEPA."""
    torch = _import_torch()
    produced = 0
    retries = 0

    while produced < clip_limit and not stop_event.is_set():
        try:
            ds = _create_stream(0, local_data=local_data)
            pending_bytes = []
            pending_keys = []

            for example in ds:
                if stop_event.is_set():
                    break

                clip_key = get_clip_key(example)

                if subset_keys and clip_key not in subset_keys:
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
                    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                        futures = [
                            pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
                            for b, k in zip(pending_bytes, pending_keys)
                        ]
                        results = [(f.result(), k) for f, k in zip(futures, pending_keys)]

                    batch_tensors = []
                    batch_keys = []
                    for tensor, key in results:
                        if tensor is not None:
                            # Shuffle frames: deterministic per clip key
                            seed = hash(key) % (2**31)
                            rng = torch.Generator()
                            rng.manual_seed(seed)
                            perm = torch.randperm(tensor.shape[0], generator=rng)
                            tensor = tensor[perm]
                            batch_tensors.append(tensor)
                            batch_keys.append(key)

                    if batch_tensors:
                        processed_list = []
                        for vt in batch_tensors:
                            processed = processor(vt, return_tensors="pt")
                            processed_list.append(processed["pixel_values_videos"])
                        batched_pixels = torch.cat(processed_list, dim=0)
                        q.put(("batch", batched_pixels, batch_keys[:]))
                        produced += len(batch_keys)

                    pending_bytes = []
                    pending_keys = []
                    retries = 0

                    if produced >= clip_limit:
                        break

            # Final partial batch
            if pending_bytes and not stop_event.is_set():
                with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                    futures = [
                        pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
                        for b, k in zip(pending_bytes, pending_keys)
                    ]
                    results = [(f.result(), k) for f, k in zip(futures, pending_keys)]

                batch_tensors = []
                batch_keys = []
                for tensor, key in results:
                    if tensor is not None:
                        seed = hash(key) % (2**31)
                        rng = torch.Generator()
                        rng.manual_seed(seed)
                        perm = torch.randperm(tensor.shape[0], generator=rng)
                        tensor = tensor[perm]
                        batch_tensors.append(tensor)
                        batch_keys.append(key)

                if batch_tensors:
                    processed_list = []
                    for vt in batch_tensors:
                        processed = processor(vt, return_tensors="pt")
                        processed_list.append(processed["pixel_values_videos"])
                    batched_pixels = torch.cat(processed_list, dim=0)
                    q.put(("batch", batched_pixels, batch_keys[:]))
                    produced += len(batch_keys)

            break  # stream exhausted

        except (ConnectionError, TimeoutError, OSError) as e:
            retries += 1
            if retries > MAX_STREAM_RETRIES:
                print(f"  ERROR: stream failed after {MAX_STREAM_RETRIES} retries: {e}")
                break
            wait = min(2 ** retries, 60)
            print(f"  WARN: stream error ({e}), retry {retries}/{MAX_STREAM_RETRIES} in {wait}s")
            time.sleep(wait)
        except Exception as e:
            print(f"  ERROR: unexpected producer error: {e}")
            import traceback
            traceback.print_exc()
            break

    q.put(("done", None, None))


def generate_shuffled_vjepa(output_dir: Path, args, clip_limit: int, subset_keys: set):
    """Task 18: Shuffled V-JEPA — same model, shuffled frame order, 1408-dim."""
    torch = _import_torch()
    from transformers import AutoModel, AutoVideoProcessor
    from m05_vjepa_embed import get_batch_embeddings

    device = "cuda"
    model_id = ENCODER_REGISTRY["vjepa_shuffled"]["model_id"]
    files = get_encoder_files("vjepa_shuffled", output_dir)
    checkpoint_file = output_dir / ".m05b_vjepa_shuffled_checkpoint.npz"

    print(f"Loading model: {model_id}")
    processor = AutoVideoProcessor.from_pretrained(model_id)
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print("FATAL: flash-attn not installed. V-JEPA 2 ViT-G requires Flash Attention 2.")
        sys.exit(1)
    model = AutoModel.from_pretrained(
        model_id, dtype=torch.float16,  # #37
        device_map="auto", attn_implementation="flash_attention_2",
    )
    model.eval()
    model = torch.compile(model)
    print(f"V-JEPA loaded for shuffled inference (dim=1408, dtype={next(model.parameters()).dtype})")

    all_embeddings, all_keys, resume_count = load_checkpoint(checkpoint_file)
    processed_keys = set(all_keys)

    # Read from: 1) CLI arg, 2) profiler JSON, 3) pipeline.yaml, 4) linear estimate
    batch_size = None
    if args.batch_size:
        batch_size = args.batch_size
    else:
        if args.profile_data:
            profile_path = Path(args.profile_data)
            if profile_path.exists():
                _profile = json.load(open(profile_path))
                if "optimal_bs" in _profile:
                    batch_size = _profile["optimal_bs"]
                    print(f"Batch size: {batch_size} (from --profile-data {profile_path})")
        if batch_size is None:
            batch_size = get_pipeline_config()["gpu"].get("inference_vjepa_bs",
                         compute_batch_sizes(gpu_vram_gb=args.gpu_mem)["vjepa"])
    if args.SANITY:
        batch_size = min(batch_size, 2)

    tmp_base = output_dir / "tmp_m05b"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)

    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_producer_shuffled_vjepa,
        args=(processor, batch_size, tmp_dir, q, stop_event,
              clip_limit, subset_keys, processed_keys, VJEPA_FRAMES_PER_CLIP,
              getattr(args, 'local_data', None)),
        daemon=True,
    )
    producer.start()

    start_time = time.time()
    last_window_count = len(all_embeddings)
    last_window_time = start_time
    pbar = tqdm(total=clip_limit, initial=resume_count,
                desc="m05b vjepa_shuffled", unit="clip")
    try:
        while True:
            try:
                msg_type, batched_pixels, batch_keys = q.get(timeout=60)
            except queue.Empty:
                print("\nProducer timeout (10 min). Saving checkpoint...")
                break

            if msg_type == "done":
                break

            emb = get_batch_embeddings(model, batched_pixels, device)

            for e, k in zip(emb, batch_keys):
                all_embeddings.append(e)
                all_keys.append(k)
                processed_keys.add(k)

            pbar.update(len(batch_keys))
            now = time.time()
            window_clips = len(all_embeddings) - last_window_count
            window_time = now - last_window_time
            throughput = window_clips / window_time if window_time > 0 else 0
            last_window_count = len(all_embeddings)
            last_window_time = now
            pbar.set_postfix({"rate": f"{throughput:.1f}/s"})

            if len(all_embeddings) % CHECKPOINT_EVERY < batch_size:
                save_checkpoint(list(all_embeddings), list(all_keys), checkpoint_file)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        stop_event.set()
    finally:
        pbar.close()
        stop_event.set()
        save_checkpoint(all_embeddings, all_keys, checkpoint_file)
        producer.join(timeout=10)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            tmp_base.rmdir()
        except OSError:
            pass

    _finalize(all_embeddings, all_keys, files, checkpoint_file,
              cache_policy=args.cache_policy,
              subset_path=args.subset, label="m05b vjepa_shuffled")
    return np.stack(all_embeddings).astype(np.float32), all_keys


# ── Shared Finalize ──────────────────────────────────────────────────

def _finalize(all_embeddings: list, all_keys: list, files: dict, checkpoint_file: Path,
              cache_policy: str, subset_path: str, label: str):
    """Save final output, clean checkpoint. cache_policy + subset_path + label all required."""
    if not all_embeddings:
        print("ERROR: No embeddings collected.")
        sys.exit(1)

    embeddings = np.stack(all_embeddings).astype(np.float32)
    # Defensive shape check (incident 2026-04-26 — see utils.config).
    verify_npy_matches_subset(embeddings, subset_path, label=label)
    files["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(files["embeddings"], embeddings)
    np.save(files["paths"], np.array(all_keys, dtype=object))

    # iter11 META-fix: gate checkpoint cleanup through --cache-policy (1=keep / 2=recompute).
    # Bug fix 2026-04-27: previously read `args.cache_policy` from outer scope (F821 — `args`
    # not defined inside _finalize); now passed explicitly by caller.
    from utils.cache_policy import guarded_delete
    guarded_delete(checkpoint_file, cache_policy, label="m05b baselines checkpoint")

    print(f"\nSaved: {files['embeddings']} ({embeddings.shape})")
    print(f"Saved: {files['paths']} ({len(all_keys)} keys)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    cleanup_temp()
    parser = argparse.ArgumentParser(
        description="Generate baseline embeddings: Random, DINOv2, CLIP, Shuffled V-JEPA")
    ALL_BASELINES = ["random", "oracle", "dinov2", "clip", "vjepa_shuffled"]
    parser.add_argument("--encoder", required=True,
                        choices=ALL_BASELINES + ["all"],
                        help="Baseline encoder to run ('all' = run all 5 sequentially)")
    parser.add_argument("--tags-json", required=True,
                        help="Path to tags.json (e.g., data/ultra_hard_3066_local/tags.json) "
                             "— required for oracle encoder; ignored by other baselines.")
    parser.add_argument("--SANITY", action="store_true", help="Process 5 clips only")
    parser.add_argument("--POC", action="store_true", help="10K subset")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (auto-computed if omitted)")
    parser.add_argument("--profile-data", type=str, default=None,
                        help="Optional profile JSON from src/utils/profile_vram.py "
                             "(e.g., outputs/profile/inference/{dinov2,vjepa2}/profile_data.json). "
                             "If omitted, falls back to pipeline.yaml gpu.inference_*_bs.")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    add_gpu_mem_arg(parser)
    # Cache-policy gate: ONLY needed for GPU baselines that produce a checkpoint
    # (.m05b_<enc>_checkpoint.npz used by `_finalize` → guarded_delete). Synthetic
    # CPU-only baselines (random, oracle) never call _finalize and never create a
    # checkpoint — recompute is <3 sec, so the prompt is pure ceremony for them.
    # See errors_N_fixes #80 for the same docstring justification applied to m08b/m06.
    from utils.cache_policy import add_cache_policy_arg, resolve_cache_policy_interactive
    add_cache_policy_arg(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    ensure_local_data(args)
    encoders_to_run = ALL_BASELINES if args.encoder == "all" else [args.encoder]

    # Bypass the cache-policy prompt when ALL requested encoders are synthetic.
    # GPU baselines (dinov2/clip/vjepa_shuffled) and "all" mode still go through
    # the interactive resolver because they create / consume checkpoint state.
    SYNTHETIC_ENCODERS = {"random", "oracle"}
    needs_cache_prompt = any(e not in SYNTHETIC_ENCODERS for e in encoders_to_run)
    if needs_cache_prompt:
        args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)
    else:
        # All-synthetic invocation — args.cache_policy is never read (random/oracle
        # paths skip _finalize). Silently set to "1" to satisfy any defensive
        # downstream access without prompting the operator for a 3-second job.
        args.cache_policy = "1"

    if args.encoder == "all":
        print(f"{'='*60}")
        print("RUNNING ALL 4 BASELINES SEQUENTIALLY")
        print(f"Order: {' → '.join(encoders_to_run)}")
        print(f"{'='*60}\n")

    for enc_idx, encoder in enumerate(encoders_to_run):
        if len(encoders_to_run) > 1:
            print(f"\n{'#'*60}")
            print(f"# [{enc_idx+1}/{len(encoders_to_run)}] {encoder}")
            print(f"{'#'*60}\n")

        _run_single_encoder(encoder, args)

        # Cleanup CUDA between encoders to prevent cross-encoder fragmentation
        if len(encoders_to_run) > 1:
            cuda_cleanup()

    if args.encoder == "all":
        print(f"\n{'='*60}")
        print("ALL 4 BASELINES COMPLETE")
        print("Next: python -u src/m06_faiss_metrics.py --encoder <enc> --FULL --subset data/subset_10k.json")
        print(f"{'='*60}")


def _run_single_encoder(encoder: str, args):
    """Run a single baseline encoder end-to-end."""
    info = ENCODER_REGISTRY[encoder]
    output_dir = get_module_output_dir("m05b_baselines", args.subset, sanity=args.SANITY, poc=args.POC)
    files = get_encoder_files(encoder, output_dir)

    print(f"{'='*60}")
    print(f"BASELINE: {encoder}")
    print(f"Model:    {info['model_id'] or 'N/A (synthetic)'}")
    print(f"Dim:      {info['dim']}")
    print(f"Type:     {info['type']}")
    print(f"Output:   {files['embeddings']}")
    print(f"{'='*60}")

    # Determine clip limit + subset
    subset_keys = load_subset(args.subset) if args.subset else set()
    if args.SANITY:
        clip_limit = get_sanity_clip_limit("embed")
    elif subset_keys:
        clip_limit = len(subset_keys)
    else:
        clip_limit = get_total_clips(local_data=getattr(args, 'local_data', None))
        if clip_limit == 0:
            print("FATAL: Cannot determine clip count. Use --subset or --local-data with manifest.json")
            sys.exit(1)

    # Random + oracle are CPU-safe synthetic baselines; others need GPU.
    if encoder == "oracle":
        # Load reference clip keys from frozen V-JEPA embeddings.paths.npy
        # (same alignment pattern as random).
        vjepa_files = get_encoder_files("vjepa_2_1_frozen", output_dir)
        if vjepa_files["paths"].exists():
            ref_keys = list(np.load(vjepa_files["paths"], allow_pickle=True))
            print(f"Using {len(ref_keys):,} clip keys from {vjepa_files['paths']}")
        else:
            # Fall back to subset_keys → list (FAISS oracle only needs path-like keys
            # for source_file alignment, no shard offsets required)
            ref_keys = sorted(subset_keys) if subset_keys else []
            if not ref_keys:
                print("FATAL: oracle needs reference paths — run frozen m05 first or pass --subset.")
                sys.exit(1)
            print(f"Falling back to {len(ref_keys):,} subset keys (frozen embeddings missing)")
        if args.SANITY:
            ref_keys = ref_keys[:get_sanity_clip_limit("embed")]
        wb_run = init_wandb("m05b", f"oracle_{'SANITY' if args.SANITY else 'POC' if args.POC else 'FULL'}",
                            config=vars(args), enabled=not args.no_wandb)
        embeddings, keys = generate_oracle(ref_keys, output_dir, args)
        log_metrics(wb_run, {"total_clips": len(keys), "embedding_dim": embeddings.shape[1]})
        log_artifact(wb_run, f"embeddings_{encoder}", str(files["embeddings"]))
        finish_wandb(wb_run)
    elif encoder == "random":
        # Load reference clip keys from V-JEPA embeddings.paths.npy
        vjepa_files = get_encoder_files("vjepa", output_dir)
        if vjepa_files["paths"].exists():
            ref_keys = list(np.load(vjepa_files["paths"], allow_pickle=True))
            print(f"Using {len(ref_keys):,} clip keys from V-JEPA embeddings.paths.npy")
        else:
            # iter13 v13 FIX-13 (2026-05-07): FAIL HARD per CLAUDE.md.
            # Previously this fabricated synthetic clip keys ("clip_000000",
            # "clip_000001", ...) which match NOTHING downstream — paired-Δ
            # m06 silently produced wrong numbers because the random_baseline
            # embeddings carried fake IDs. Random baseline EXISTS to anchor
            # the BCa CI on the SAME clip-key universe as V-JEPA, so missing
            # paths.npy is a hard prerequisite failure, not a recoverable warn.
            print(f"FATAL: {vjepa_files['paths']} not found.")
            print("  m05b random_baseline requires V-JEPA's clip-key set as")
            print("  the reference (paired-Δ across encoders demands identical")
            print("  clip-key universe). Run m05_vjepa_embed.py first.")
            sys.exit(1)

        if args.SANITY:
            ref_keys = ref_keys[:get_sanity_clip_limit("embed")]

        wb_run = init_wandb("m05b", f"random_{'SANITY' if args.SANITY else 'POC' if args.POC else 'FULL'}",
                            config=vars(args), enabled=not args.no_wandb)
        embeddings, keys = generate_random(ref_keys, output_dir, args)
        log_metrics(wb_run, {"total_clips": len(keys), "embedding_dim": embeddings.shape[1]})
        log_artifact(wb_run, f"embeddings_{encoder}", str(files["embeddings"]))
        finish_wandb(wb_run)
    else:
        check_gpu()
        mode = f"{encoder}_{'SANITY' if args.SANITY else 'POC' if args.POC else 'FULL'}"
        wb_run = init_wandb("m05b", mode, config=vars(args), enabled=not args.no_wandb)

        if encoder == "dinov2":
            embeddings, keys = generate_dinov2(output_dir, args, clip_limit, subset_keys)
        elif encoder == "clip":
            embeddings, keys = generate_clip(output_dir, args, clip_limit, subset_keys)
        elif encoder == "vjepa_shuffled":
            embeddings, keys = generate_shuffled_vjepa(output_dir, args, clip_limit, subset_keys)

        log_metrics(wb_run, {"total_clips": len(keys), "embedding_dim": embeddings.shape[1]})
        log_artifact(wb_run, f"embeddings_{encoder}", str(files["embeddings"]))
        finish_wandb(wb_run)

    print(f"\n=== BASELINE COMPLETE: {encoder} ===")
    print(f"Clips:     {len(keys):,}")
    print(f"Dim:       {embeddings.shape[1]}")
    print(f"Next step: python -u src/m06_faiss_metrics.py --encoder {encoder} --FULL --subset data/subset_10k.json")

    # Force exit: torch.compile + CUDA atexit cleanup deadlocks on futex_wait_queue
    # (hangs indefinitely at 0% GPU after all output files are saved)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
