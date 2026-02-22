"""
Tag video clips with Qwen3-VL-8B via vLLM batched inference + HF WebDataset streaming.
Streams clips from HF TAR shards (no local clips needed). Saves tags.json with checkpoint/resume.

USAGE:
    python -u src/m04_qwen_tag.py --SANITY 2>&1 | tee logs/m04_qwen_tag_sanity.log
    python -u src/m04_qwen_tag.py --FULL 2>&1 | tee logs/m04_qwen_tag_full.log
"""
import argparse
import gc
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

# ── Fix Issue 2: Force vLLM V0 engine (V1 leaks CPU RAM) ─────────────────
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    TAGS_FILE, QWEN_MODEL_ID, TAG_TAXONOMY_JSON,
    HF_DATASET_REPO, OUTPUTS_DIR,
    check_gpu, check_output_exists,
)

# ── vLLM + Qwen imports (GPU-only) ──────────────────────────────────────────
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    HAS_VLLM = True
except ImportError as e:
    HAS_VLLM = False
    print(f"WARNING: vLLM/Qwen dependencies not available: {e}")
    print("Install: pip install vllm qwen-vl-utils transformers")

# ── HF datasets (streaming) ─────────────────────────────────────────────────
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("WARNING: datasets not available. Install: pip install datasets")

# ── HF auth ──────────────────────────────────────────────────────────────────
_hf_token = None
try:
    from dotenv import load_dotenv
    load_dotenv()
    _hf_token = os.getenv("HF_TOKEN")
    if _hf_token:
        from huggingface_hub import login
        login(token=_hf_token, add_to_git_credential=False)
        print("HuggingFace: Authenticated with HF_TOKEN")
    else:
        print("WARNING: HF_TOKEN not found in .env")
except (ImportError, OSError) as e:
    print(f"WARNING: HF auth skipped ({e})")


# ── Tag taxonomy (single source of truth) ────────────────────────────────────

def load_taxonomy() -> dict:
    with open(TAG_TAXONOMY_JSON, 'r') as f:
        taxonomy = json.load(f)
    return {k: v for k, v in taxonomy.items() if not k.startswith("_")}

TAXONOMY = load_taxonomy()


def build_tag_prompt(taxonomy: dict) -> str:
    lines = []
    multi_fields = []
    for field, spec in taxonomy.items():
        values = spec["values"]
        if spec["type"] == "single":
            lines.append(f'  "{field}": "{"|".join(values)}"')
        else:
            multi_fields.append(field)
            lines.append(f'  "{field}": ["subset of: {", ".join(values)}"]')

    json_block = "{\n" + ",\n".join(lines) + "\n}"
    if multi_fields:
        joined = '" and "'.join(multi_fields)
        multi_note = f'For "{joined}": list ALL that apply.'
    else:
        multi_note = ""

    return (
        f"Analyze this Indian street video clip. Output ONLY a JSON object with these fields:\n\n"
        f"{json_block}\n\n"
        f"{multi_note}\n"
        f"For all other fields: pick exactly ONE value.\n"
        f"Output ONLY the JSON, no explanation."
    )

TAG_PROMPT = build_tag_prompt(TAXONOMY)

# ── Config ───────────────────────────────────────────────────────────────────
VLLM_BATCH_SIZE = 8
CHECKPOINT_EVERY = 500               # Issue 8: was 1000, now saves more often
ENGINE_RESTART_EVERY = 10_000         # Issue 1: restart vLLM process every N clips
MAX_STREAM_RETRIES = 5                # Issue 5: network retry limit
PREPROCESS_WORKERS = 4                # Issue 3: parallel preprocessing threads
PREFETCH_QUEUE_SIZE = 2               # Issue 9: pipeline buffer depth
TOTAL_CLIPS = 115_687                 # known total from m03


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_json_output(output_text: str) -> dict | None:
    try:
        start = output_text.find('{')
        end = output_text.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(output_text[start:end])
    except json.JSONDecodeError:
        pass
    return None


def get_dummy_tag() -> dict:
    return {field: spec["default"] for field, spec in TAXONOMY.items()}


def validate_mp4(path: str) -> bool:
    """Issue 6: Pre-validate MP4 before passing to VLM."""
    try:
        if os.path.getsize(path) < 1024:  # < 1KB = definitely broken
            return False
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
            cap.release()
            return ok
        except ImportError:
            return True  # cv2 not available, trust size check
    except OSError:
        return False


# ── Checkpoint (Issue 8: atomic writes + safe load) ──────────────────────────

def save_checkpoint(all_tags: list, tags_file: Path):
    """Atomic write: write to .tmp, then os.replace (POSIX atomic)."""
    tags_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tags_file.with_suffix(".json.tmp")
    with open(tmp_file, 'w') as f:
        json.dump(all_tags, f)
    os.replace(tmp_file, tags_file)


def load_checkpoint(tags_file: Path) -> tuple[list, int]:
    """Safe load with corruption recovery from .tmp backup."""
    if not tags_file.exists():
        return [], 0
    try:
        with open(tags_file) as f:
            all_tags = json.load(f)
        if isinstance(all_tags, list):
            return all_tags, len(all_tags)
    except (json.JSONDecodeError, OSError):
        pass
    # Try .tmp backup
    tmp_file = tags_file.with_suffix(".json.tmp")
    if tmp_file.exists():
        try:
            with open(tmp_file) as f:
                all_tags = json.load(f)
            if isinstance(all_tags, list):
                print(f"  Recovered {len(all_tags)} tags from .tmp backup")
                return all_tags, len(all_tags)
        except (json.JSONDecodeError, OSError):
            pass
    print("  WARN: checkpoint corrupted, starting fresh")
    return [], 0


# ── Preprocessing (Issues 3, 6, 7) ──────────────────────────────────────────

def preprocess_one(example: dict, processor, tmp_dir: str) -> dict | None:
    """Preprocess single clip: tempfile → validate → process_vision_info → cleanup."""
    tmp_path = None
    try:
        mp4_data = example["mp4"]
        mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
        key = example.get("__key__", "unknown")

        # Issue 7: write to custom tmp_dir, not system /tmp
        tmp_path = os.path.join(tmp_dir, f"{key}.mp4")
        with open(tmp_path, "wb") as f:
            f.write(mp4_bytes)

        # Issue 6: validate before VLM
        if not validate_mp4(tmp_path):
            print(f"  WARN: invalid mp4 {key}, skipping")
            return None

        messages = [{"role": "user", "content": [
            {"type": "video", "video": tmp_path, "max_pixels": 360 * 420, "fps": 1.0},
            {"type": "text", "text": TAG_PROMPT},
        ]}]

        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        mm_data = {}
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
    except Exception as e:
        print(f"  WARN: preprocess failed ({example.get('__key__', '?')}): {e}")
        return None
    finally:
        # Issue 7: always cleanup tempfile
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def preprocess_batch(batch: list, processor, tmp_dir: str) -> list:
    """Issue 3: Parallel preprocessing via ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as pool:
        return list(pool.map(
            lambda ex: preprocess_one(ex, processor, tmp_dir), batch
        ))


# ── Inference ────────────────────────────────────────────────────────────────

def tag_batch_vllm(llm, sampling_params, preprocessed: list, batch: list) -> list:
    """Run vLLM batched inference, merge results with metadata."""
    valid_indices = {i for i, p in enumerate(preprocessed) if p is not None}
    valid_inputs = [preprocessed[i] for i in sorted(valid_indices)]

    outputs = llm.generate(valid_inputs, sampling_params) if valid_inputs else []

    results = []
    out_i = 0
    for i, example in enumerate(batch):
        metadata = example["json"] if isinstance(example.get("json"), dict) else {}
        key = example.get("__key__", "")

        if i in valid_indices and out_i < len(outputs):
            tags = parse_json_output(outputs[out_i].outputs[0].text)
            if tags is None:
                tags = get_dummy_tag()
            out_i += 1
        else:
            tags = get_dummy_tag()

        results.append({"__key__": key, **metadata, **tags})
    return results


# ── Pipeline: Producer/Consumer (Issues 5, 9) ───────────────────────────────

def _create_stream(skip_count: int):
    """Create HF dataset stream, skip to resume position."""
    ds = load_dataset(HF_DATASET_REPO, split="train", streaming=True)
    ds = ds.decode(False)
    if skip_count > 0:
        ds = ds.skip(skip_count)
    return ds


def _producer_thread(start_from, processor, batch_size, tmp_dir,
                     q, stop_event, clip_limit):
    """
    Issue 5: Retryable HF streaming.
    Issue 9: Runs in background — preprocesses while GPU infers.
    """
    produced = 0
    retries = 0

    while produced < clip_limit and not stop_event.is_set():
        try:
            ds = _create_stream(start_from + produced)
            batch = []

            for example in ds:
                if stop_event.is_set():
                    break
                batch.append(example)

                if len(batch) >= batch_size:
                    preprocessed = preprocess_batch(batch, processor, tmp_dir)
                    q.put(("batch", batch, preprocessed))
                    produced += len(batch)
                    batch = []
                    retries = 0

                    if produced >= clip_limit:
                        break

            # Final partial batch
            if batch and not stop_event.is_set():
                preprocessed = preprocess_batch(batch, processor, tmp_dir)
                q.put(("batch", batch, preprocessed))
                produced += len(batch)

            break  # stream exhausted normally

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
            break

    q.put(("done", None, None))


def stream_and_tag(llm, processor, sampling_params, args,
                   start_from: int = 0, clip_limit: int = 0) -> list:
    """Stream from HF, preprocess in background thread, infer on GPU."""
    print(f"\nStreaming from: {HF_DATASET_REPO}")
    print(f"  start_from={start_from:,}  clip_limit={clip_limit:,}")

    if start_from > 0:
        print(f"  (skip happens inside producer thread)")

    # Load existing checkpoint
    all_tags, _ = load_checkpoint(TAGS_FILE)

    # Issue 7: project-local tmp dir (not /tmp which may be tmpfs)
    tmp_base = OUTPUTS_DIR / "tmp_m04"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)

    # Issue 9: prefetch pipeline — producer preprocesses while GPU infers
    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_producer_thread,
        args=(start_from, processor, args.batch_size, tmp_dir,
              q, stop_event, clip_limit),
        daemon=True,
    )
    producer.start()

    clips_this_run = 0
    start_time = time.time()

    try:
        while True:
            msg_type, batch, preprocessed = q.get(timeout=600)  # 10 min timeout
            if msg_type == "done":
                break

            results = tag_batch_vllm(llm, sampling_params, preprocessed, batch)
            all_tags.extend(results)
            clips_this_run += len(results)

            # Progress
            elapsed = time.time() - start_time
            rate = clips_this_run / elapsed if elapsed > 0 else 0
            remaining = clip_limit - clips_this_run
            eta_min = remaining / rate / 60 if rate > 0 else 0
            last_scene = results[-1].get("scene_type", "?")[:10]
            print(f"  [{start_from + clips_this_run:,}/{start_from + clip_limit:,}] "
                  f"{rate:.2f} clips/s | ETA {eta_min:.0f} min | scene={last_scene}")

            # Issue 8: checkpoint with atomic write
            if clips_this_run % CHECKPOINT_EVERY < args.batch_size:
                save_checkpoint(all_tags, TAGS_FILE)
                print(f"  -- checkpoint: {len(all_tags):,} tags saved --")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        stop_event.set()
    except queue.Empty:
        print("\nProducer timeout (10 min). Saving checkpoint...")
        stop_event.set()
    except Exception as e:
        print(f"\nInference error: {e}. Saving checkpoint...")
        stop_event.set()
    finally:
        save_checkpoint(all_tags, TAGS_FILE)
        stop_event.set()
        producer.join(timeout=10)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if clips_this_run > 0:
        elapsed = time.time() - start_time
        print(f"\nSegment done: {clips_this_run:,} clips in {elapsed:.0f}s "
              f"({clips_this_run/elapsed:.2f} clips/s)")

    return all_tags


# ── Issue 1: Subprocess-based engine restart ─────────────────────────────────

def orchestrator_main(args):
    """
    Spawn worker subprocesses every ENGINE_RESTART_EVERY clips.
    Each worker loads vLLM, processes a segment, exits — fully releasing GPU memory.
    """
    total_clips = 20 if args.SANITY else TOTAL_CLIPS

    all_tags, skip_count = load_checkpoint(TAGS_FILE)
    if skip_count >= total_clips:
        print(f"Already complete: {skip_count:,}/{total_clips:,} clips tagged")
        generate_plot(all_tags)
        return

    if skip_count > 0:
        print(f"Resuming from checkpoint: {skip_count:,}/{total_clips:,} clips")

    segment_idx = 0
    while skip_count < total_clips:
        segment_size = min(ENGINE_RESTART_EVERY, total_clips - skip_count)
        segment_idx += 1
        print(f"\n{'='*60}")
        print(f"WORKER {segment_idx}: clips {skip_count:,} → {skip_count + segment_size:,}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "-u", os.path.abspath(__file__),
            "--_worker",
            "--start-from", str(skip_count),
            "--process-count", str(segment_size),
            "--batch-size", str(args.batch_size),
        ]
        if args.SANITY:
            cmd.append("--SANITY")
        if args.FULL:
            cmd.append("--FULL")

        result = subprocess.run(cmd)

        new_tags, new_count = load_checkpoint(TAGS_FILE)
        if new_count > skip_count:
            skip_count = new_count
            all_tags = new_tags
            print(f"Worker done. Progress: {skip_count:,}/{total_clips:,}")
        elif result.returncode != 0:
            print(f"Worker failed (exit {result.returncode}). Resume with same command.")
            break
        else:
            skip_count = total_clips  # stream exhausted
            all_tags = new_tags

    print(f"\n=== TAGGING COMPLETE ===")
    print(f"Saved: {TAGS_FILE}")
    print(f"Total clips tagged: {len(all_tags):,}")
    generate_plot(all_tags)


def worker_main(args):
    """Worker subprocess: load vLLM, process segment, exit (GPU memory freed)."""
    check_gpu()

    print(f"\nLoading vLLM: {QWEN_MODEL_ID}")
    llm = LLM(
        model=QWEN_MODEL_ID,
        max_model_len=4096,              # Issue 4: lowered from 16384
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1},
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    sampling_params = SamplingParams(max_tokens=512, temperature=0.1)
    print(f"Model loaded | batch_size={args.batch_size} | segment={args.process_count:,}")

    stream_and_tag(
        llm, processor, sampling_params, args,
        start_from=args.start_from,
        clip_limit=args.process_count,
    )

    del llm, processor
    gc.collect()
    print("Worker exiting (GPU memory will be released).")


# ── Dummy mode ───────────────────────────────────────────────────────────────

def stream_and_tag_dummy(args) -> list:
    """Dummy tagging (no GPU). Streams from HF, assigns default tag values."""
    if not HAS_DATASETS:
        print("ERROR: datasets library required. pip install datasets")
        sys.exit(1)

    print(f"\nStreaming from: {HF_DATASET_REPO} (dummy mode)")
    ds = load_dataset(HF_DATASET_REPO, split="train", streaming=True)
    ds = ds.decode(False)

    clip_limit = 20 if args.SANITY else 100
    ds = ds.take(clip_limit)
    print(f"{'SANITY' if args.SANITY else 'FULL'} MODE (dummy): {clip_limit} clips")

    all_tags = []
    for example in ds:
        metadata = example["json"] if isinstance(example.get("json"), dict) else {}
        key = example.get("__key__", "")
        tags = get_dummy_tag()
        all_tags.append({"__key__": key, **metadata, **tags})

        if len(all_tags) % 10 == 0:
            print(f"  [dummy] {len(all_tags)}/{clip_limit} clips tagged")

    return all_tags


# ── Plot ─────────────────────────────────────────────────────────────────────

def generate_plot(all_tags: list):
    """Generate scene distribution bar chart (.png + .pdf)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plot")
        return

    scene_counts = {}
    for t in all_tags:
        st = t.get("scene_type", "unknown")
        scene_counts[st] = scene_counts.get(st, 0) + 1

    print(f"\nScene type distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: -x[1]):
        print(f"  {scene}: {count}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    scenes = sorted(scene_counts.keys(), key=lambda x: scene_counts[x], reverse=True)
    counts = [scene_counts[s] for s in scenes]
    colors = plt.cm.tab10(range(len(scenes)))

    bars = ax.bar(scenes, counts, color=colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Scene Type')
    ax.set_ylabel('Number of Clips')
    ax.set_title(f'Qwen3-VL Scene Type Distribution (n={len(all_tags)} clips)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plot_png = OUTPUTS_DIR / "m04_scene_distribution.png"
    plot_pdf = OUTPUTS_DIR / "m04_scene_distribution.pdf"
    plt.savefig(plot_png, dpi=150)
    plt.savefig(plot_pdf)
    plt.close()
    print(f"Saved: {plot_png}")
    print(f"Saved: {plot_pdf}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tag video clips with Qwen3-VL-8B via vLLM (streams from HF WebDataset)")
    parser.add_argument("--SANITY", action="store_true", help="Process 20 clips only")
    parser.add_argument("--FULL", action="store_true", help="Process all ~115k clips")
    parser.add_argument("--dummy", action="store_true", help="Use dummy tags (no GPU/model)")
    parser.add_argument("--batch-size", type=int, default=VLLM_BATCH_SIZE,
                        help=f"vLLM batch size (default: {VLLM_BATCH_SIZE})")
    # Internal args for subprocess worker (Issue 1)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start-from", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--process-count", type=int, default=ENGINE_RESTART_EVERY,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Worker mode (spawned by orchestrator)
    if args._worker:
        if not HAS_VLLM:
            print("ERROR: vLLM required for worker mode")
            sys.exit(1)
        worker_main(args)
        return

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    # Check existing complete tags.json
    total = 20 if args.SANITY else TOTAL_CLIPS
    if TAGS_FILE.exists() and not args.dummy:
        all_tags, count = load_checkpoint(TAGS_FILE)
        if count >= total:
            if not check_output_exists([TAGS_FILE], "tags"):
                print(f"Loaded {count:,} cached tags (complete)")
                generate_plot(all_tags)
                return

    # Dummy mode
    if args.dummy or not HAS_VLLM:
        if not args.dummy:
            print("vLLM not available, falling back to dummy tags")
        all_tags = stream_and_tag_dummy(args)
        save_checkpoint(all_tags, TAGS_FILE)
        print(f"\n=== TAGGING COMPLETE (dummy) ===")
        print(f"Saved: {TAGS_FILE}")
        print(f"Total clips tagged: {len(all_tags):,}")
        generate_plot(all_tags)
    else:
        # GPU mode: orchestrator spawns worker subprocesses (Issue 1)
        orchestrator_main(args)


if __name__ == "__main__":
    main()
