"""
VLM tagging with bake-off: 3 backends (Qwen3-VL via vLLM, VideoLLaMA3, Keye-VL via transformers).
Orchestrator/worker pattern for VRAM management. HF WebDataset streaming with checkpoint/resume.

USAGE:
    python -u src/m04_vlm_tag.py --model qwen --SANITY 2>&1 | tee logs/m04_sanity_qwen.log
    python -u src/m04_vlm_tag.py --model qwen --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_qwen_poc.log
    python -u src/m04_vlm_tag.py --model videollama --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_videollama_poc.log
    python -u src/m04_vlm_tag.py --model keye --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_keye_poc.log
    python -u src/m04_vlm_tag.py --model qwen --FULL --subset data/subset_10k.json 2>&1 | tee logs/m04_full_qwen_poc.log
    python -u src/m04_vlm_tag.py --model qwen --FULL 2>&1 | tee logs/m04_full_qwen.log
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
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# ── Fix vLLM V0 engine leak (applies only when vLLM is used) ─────────────
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    TAGS_FILE, TAG_TAXONOMY_JSON, HF_DATASET_REPO, OUTPUTS_DIR,
    VLM_MODELS, BAKEOFF_CLIP_COUNT, BAKEOFF_DIR, OUTPUTS_POC_DIR,
    check_gpu, check_output_exists, load_subset, add_subset_arg,
)
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, log_artifact, finish_wandb,
)

# ── HF datasets (streaming) ──────────────────────────────────────────────
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("WARNING: datasets not available. Install: pip install datasets")

# ── HF auth ──────────────────────────────────────────────────────────────
_hf_token = None
try:
    from dotenv import load_dotenv
    load_dotenv()
    _hf_token = os.getenv("HF_TOKEN")
    if _hf_token:
        from huggingface_hub import login
        login(token=_hf_token, add_to_git_credential=False)
        print("HuggingFace: Authenticated")
    else:
        print("WARNING: HF_TOKEN not found in .env")
except (ImportError, OSError) as e:
    print(f"WARNING: HF auth skipped ({e})")


# ═════════════════════════════════════════════════════════════════════════
# TAG TAXONOMY + PROMPT (shared across all backends)
# ═════════════════════════════════════════════════════════════════════════

def load_taxonomy() -> dict:
    with open(TAG_TAXONOMY_JSON, 'r') as f:
        taxonomy = json.load(f)
    return {k: v for k, v in taxonomy.items() if not k.startswith("_")}

TAXONOMY = load_taxonomy()


def build_tag_prompt(taxonomy: dict) -> str:
    lines = []
    conf_lines = []
    multi_fields = []
    for field, spec in taxonomy.items():
        values = spec["values"]
        if spec["type"] == "single":
            lines.append(f'  "{field}": "{"|".join(values)}"')
        else:
            multi_fields.append(field)
            lines.append(f'  "{field}": ["subset of: {", ".join(values)}"]')
        conf_lines.append(f'  "confidence_{field}": 0.0-1.0')

    json_block = "{\n" + ",\n".join(lines) + ",\n" + ",\n".join(conf_lines) + "\n}"
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
        f"For confidence_* fields: output your confidence as a float in [0.0, 1.0].\n"
        f"Output ONLY the JSON, no explanation."
    )

TAG_PROMPT = build_tag_prompt(TAXONOMY)


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
    """Pre-validate MP4 before passing to VLM."""
    try:
        if os.path.getsize(path) < 1024:
            return False
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
            cap.release()
            return ok
        except ImportError:
            return True
    except OSError:
        return False


# ═════════════════════════════════════════════════════════════════════════
# VLM BACKEND ABC
# ═════════════════════════════════════════════════════════════════════════

PREPROCESS_WORKERS = 4


class VLMBackend(ABC):
    """Abstract base class for VLM tagging backends."""

    def __init__(self, model_name: str, model_id: str):
        self.model_name = model_name
        self.model_id = model_id

    @abstractmethod
    def load_model(self) -> None:
        """Load model + processor into GPU memory."""

    @abstractmethod
    def preprocess_one(self, example: dict, tmp_dir: str) -> dict | None:
        """Preprocess a single clip. Returns backend-specific dict or None on failure."""

    def preprocess_batch(self, batch: list, tmp_dir: str) -> list:
        """Parallel preprocessing via ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as pool:
            return list(pool.map(lambda ex: self.preprocess_one(ex, tmp_dir), batch))

    @abstractmethod
    def generate_batch(self, preprocessed: list, batch: list) -> list:
        """Run inference on preprocessed batch. Returns list of tag dicts (one per clip)."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release GPU memory."""


# ═════════════════════════════════════════════════════════════════════════
# BACKEND: Qwen3-VL-8B (vLLM batched inference)
# ═════════════════════════════════════════════════════════════════════════

class QwenBackend(VLMBackend):
    """Qwen3-VL-8B via vLLM. Fastest backend (batched GPU inference)."""

    def __init__(self):
        super().__init__("qwen", VLM_MODELS["qwen"])
        self.llm = None
        self.processor = None
        self.sampling_params = None

    def load_model(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        print(f"Loading vLLM: {self.model_id}")
        self.llm = LLM(
            model=self.model_id,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            limit_mm_per_prompt={"video": 1},
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.sampling_params = SamplingParams(max_tokens=512, temperature=0.1)
        print(f"Qwen loaded via vLLM")

    def preprocess_one(self, example: dict, tmp_dir: str) -> dict | None:
        from qwen_vl_utils import process_vision_info

        tmp_path = None
        try:
            mp4_data = example["mp4"]
            mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
            key = example.get("__key__", "unknown")

            tmp_path = os.path.join(tmp_dir, f"{key}.mp4")
            with open(tmp_path, "wb") as f:
                f.write(mp4_bytes)

            if not validate_mp4(tmp_path):
                return None

            messages = [{"role": "user", "content": [
                {"type": "video", "video": tmp_path, "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": TAG_PROMPT},
            ]}]

            prompt = self.processor.apply_chat_template(
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
            print(f"  WARN: qwen preprocess failed ({example.get('__key__', '?')}): {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def generate_batch(self, preprocessed: list, batch: list) -> list:
        valid_indices = {i for i, p in enumerate(preprocessed) if p is not None}
        valid_inputs = [preprocessed[i] for i in sorted(valid_indices)]

        outputs = self.llm.generate(valid_inputs, self.sampling_params) if valid_inputs else []

        results = []
        out_i = 0
        for i, example in enumerate(batch):
            if i in valid_indices and out_i < len(outputs):
                tags = parse_json_output(outputs[out_i].outputs[0].text)
                if tags is None:
                    tags = get_dummy_tag()
                out_i += 1
            else:
                tags = get_dummy_tag()
            results.append(tags)
        return results

    def cleanup(self):
        del self.llm, self.processor
        self.llm = self.processor = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════
# BACKEND: VideoLLaMA3-7B (transformers, sequential inference)
# ═════════════════════════════════════════════════════════════════════════

class VideoLLaMA3Backend(VLMBackend):
    """VideoLLaMA3-7B via transformers. Best MLVU + PerceptionTest, SigLIP encoder."""

    def __init__(self):
        super().__init__("videollama", VLM_MODELS["videollama"])
        self.model = None
        self.processor = None

    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading transformers: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        print(f"VideoLLaMA3 loaded via transformers (FA2)")

    def preprocess_one(self, example: dict, tmp_dir: str) -> dict | None:
        tmp_path = None
        try:
            mp4_data = example["mp4"]
            mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
            key = example.get("__key__", "unknown")

            tmp_path = os.path.join(tmp_dir, f"{key}.mp4")
            with open(tmp_path, "wb") as f:
                f.write(mp4_bytes)

            if not validate_mp4(tmp_path):
                return None

            return {"video_path": tmp_path, "key": key}
        except Exception as e:
            print(f"  WARN: videollama preprocess failed ({example.get('__key__', '?')}): {e}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return None

    def generate_batch(self, preprocessed: list, batch: list) -> list:
        import torch

        results = []
        for i, example in enumerate(batch):
            pp = preprocessed[i]
            if pp is None:
                results.append(get_dummy_tag())
                continue

            video_path = pp["video_path"]
            try:
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": 1.0, "max_frames": 64}},
                    {"type": "text", "text": TAG_PROMPT},
                ]}]

                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=text, videos=[video_path],
                    return_tensors="pt", padding=True
                ).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=512, temperature=0.1, do_sample=True
                    )
                # Decode only generated tokens (skip input)
                generated = output_ids[0][inputs["input_ids"].shape[1]:]
                response = self.processor.decode(generated, skip_special_tokens=True)

                tags = parse_json_output(response)
                if tags is None:
                    tags = get_dummy_tag()
                results.append(tags)

            except Exception as e:
                print(f"  WARN: videollama inference failed ({pp['key']}): {e}")
                results.append(get_dummy_tag())
            finally:
                if os.path.exists(video_path):
                    try:
                        os.unlink(video_path)
                    except OSError:
                        pass

        return results

    def cleanup(self):
        del self.model, self.processor
        self.model = self.processor = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════
# BACKEND: Keye-VL-1.5-8B (transformers, sequential inference)
# ═════════════════════════════════════════════════════════════════════════

class KeyeVLBackend(VLMBackend):
    """Keye-VL-1.5-8B via transformers. Highest VideoMME (beats GPT-4o), SlowFast encoding."""

    def __init__(self):
        super().__init__("keye", VLM_MODELS["keye"])
        self.model = None
        self.processor = None

    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading transformers: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        print(f"Keye-VL loaded via transformers (FA2)")

    def preprocess_one(self, example: dict, tmp_dir: str) -> dict | None:
        tmp_path = None
        try:
            mp4_data = example["mp4"]
            mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
            key = example.get("__key__", "unknown")

            tmp_path = os.path.join(tmp_dir, f"{key}.mp4")
            with open(tmp_path, "wb") as f:
                f.write(mp4_bytes)

            if not validate_mp4(tmp_path):
                return None

            return {"video_path": tmp_path, "key": key}
        except Exception as e:
            print(f"  WARN: keye preprocess failed ({example.get('__key__', '?')}): {e}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return None

    def generate_batch(self, preprocessed: list, batch: list) -> list:
        import torch

        results = []
        for i, example in enumerate(batch):
            pp = preprocessed[i]
            if pp is None:
                results.append(get_dummy_tag())
                continue

            video_path = pp["video_path"]
            try:
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": TAG_PROMPT},
                ]}]

                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=text, videos=[video_path],
                    return_tensors="pt", padding=True
                ).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=512, temperature=0.1, do_sample=True
                    )
                generated = output_ids[0][inputs["input_ids"].shape[1]:]
                response = self.processor.decode(generated, skip_special_tokens=True)

                tags = parse_json_output(response)
                if tags is None:
                    tags = get_dummy_tag()
                results.append(tags)

            except Exception as e:
                print(f"  WARN: keye inference failed ({pp['key']}): {e}")
                results.append(get_dummy_tag())
            finally:
                if os.path.exists(video_path):
                    try:
                        os.unlink(video_path)
                    except OSError:
                        pass

        return results

    def cleanup(self):
        del self.model, self.processor
        self.model = self.processor = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════
# BACKEND REGISTRY
# ═════════════════════════════════════════════════════════════════════════

BACKENDS = {
    "qwen": QwenBackend,
    "videollama": VideoLLaMA3Backend,
    "keye": KeyeVLBackend,
}


# ═════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════

VLLM_BATCH_SIZE = 8
TRANSFORMERS_BATCH_SIZE = 4          # smaller for non-vLLM backends
CHECKPOINT_EVERY = 500
ENGINE_RESTART_EVERY = 10_000
MAX_STREAM_RETRIES = 5
PREFETCH_QUEUE_SIZE = 2
TOTAL_CLIPS = 115_687
PROMPT_VERSION = "v1.0"


def get_batch_size(model_name: str, override: int = None) -> int:
    if override:
        return override
    return VLLM_BATCH_SIZE if model_name == "qwen" else TRANSFORMERS_BATCH_SIZE


# ═════════════════════════════════════════════════════════════════════════
# OUTPUT PATH LOGIC
# ═════════════════════════════════════════════════════════════════════════

def get_tags_file(model_name: str, is_bakeoff: bool, subset_path: str = None) -> Path:
    """Determine output tags file based on mode."""
    if is_bakeoff:
        return BAKEOFF_DIR / f"tags_{model_name}.json"
    elif subset_path:
        return OUTPUTS_POC_DIR / "tags.json"
    else:
        return TAGS_FILE


# ═════════════════════════════════════════════════════════════════════════
# SUBSET FILTERING
# ═════════════════════════════════════════════════════════════════════════

def get_clip_key(example: dict) -> str:
    """Reconstruct clip key from HF WebDataset example metadata."""
    meta = example.get("json", {})
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta) if meta else {}
    section = meta.get("section", "")
    video_id = meta.get("video_id", "")
    source_file = meta.get("source_file", "")
    return f"{section}/{video_id}/{source_file}"


# ═════════════════════════════════════════════════════════════════════════
# CHECKPOINT (atomic writes + safe load)
# ═════════════════════════════════════════════════════════════════════════

def save_checkpoint(all_tags: list, tags_file: Path):
    tags_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tags_file.with_suffix(".json.tmp")
    with open(tmp_file, 'w') as f:
        json.dump(all_tags, f)
    os.replace(tmp_file, tags_file)


def load_checkpoint(tags_file: Path) -> tuple[list, int]:
    if not tags_file.exists():
        return [], 0
    try:
        with open(tags_file) as f:
            all_tags = json.load(f)
        if isinstance(all_tags, list):
            return all_tags, len(all_tags)
    except (json.JSONDecodeError, OSError):
        pass
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


# ═════════════════════════════════════════════════════════════════════════
# PROVENANCE (per-clip metadata appended after inference)
# ═════════════════════════════════════════════════════════════════════════

def add_provenance(tags: dict, example: dict, model_id: str) -> dict:
    """Merge metadata + tags + provenance into final clip record."""
    meta = example.get("json", {})
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta) if meta else {}
    key = example.get("__key__", "")

    return {
        "__key__": key,
        **meta,
        **tags,
        "_model": model_id,
        "_prompt_version": PROMPT_VERSION,
        "_tagged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ═════════════════════════════════════════════════════════════════════════
# PIPELINE: Producer / Consumer (backend-agnostic)
# ═════════════════════════════════════════════════════════════════════════

def _create_stream(skip_count: int):
    ds = load_dataset(HF_DATASET_REPO, split="train", streaming=True)
    ds = ds.decode(False)
    if skip_count > 0:
        ds = ds.skip(skip_count)
    return ds


def _producer_thread(backend, start_from, batch_size, tmp_dir,
                     q, stop_event, clip_limit, subset_keys):
    """
    Background thread: streams from HF, filters by subset, preprocesses batches.
    Puts (batch, preprocessed) onto queue for consumer.
    """
    produced = 0
    retries = 0
    skipped = 0

    while produced < clip_limit and not stop_event.is_set():
        try:
            ds = _create_stream(start_from + produced + skipped)
            batch = []

            for example in ds:
                if stop_event.is_set():
                    break

                # Subset filtering
                if subset_keys:
                    clip_key = get_clip_key(example)
                    if clip_key not in subset_keys:
                        skipped += 1
                        continue

                batch.append(example)

                if len(batch) >= batch_size:
                    preprocessed = backend.preprocess_batch(batch, tmp_dir)
                    q.put(("batch", batch, preprocessed))
                    produced += len(batch)
                    batch = []
                    retries = 0

                    if produced >= clip_limit:
                        break

            # Final partial batch
            if batch and not stop_event.is_set():
                preprocessed = backend.preprocess_batch(batch, tmp_dir)
                q.put(("batch", batch, preprocessed))
                produced += len(batch)

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
            break

    q.put(("done", None, None))


def stream_and_tag(backend: VLMBackend, args,
                   start_from: int, clip_limit: int,
                   tags_file: Path, subset_keys: set,
                   wb_run=None) -> list:
    """Stream from HF, preprocess in background, infer on GPU. Backend-agnostic."""
    print(f"\nStreaming from: {HF_DATASET_REPO}")
    print(f"  backend={backend.model_name}  start_from={start_from:,}  clip_limit={clip_limit:,}")
    if subset_keys:
        print(f"  [POC] filtering to {len(subset_keys):,} subset clips")

    all_tags, _ = load_checkpoint(tags_file)

    tmp_base = OUTPUTS_DIR / "tmp_m04"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)

    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    batch_size = get_batch_size(backend.model_name, args.batch_size)

    producer = threading.Thread(
        target=_producer_thread,
        args=(backend, start_from, batch_size, tmp_dir,
              q, stop_event, clip_limit, subset_keys),
        daemon=True,
    )
    producer.start()

    clips_this_run = 0
    start_time = time.time()

    try:
        while True:
            msg_type, batch, preprocessed = q.get(timeout=600)
            if msg_type == "done":
                break

            tag_list = backend.generate_batch(preprocessed, batch)

            for example, tags in zip(batch, tag_list):
                record = add_provenance(tags, example, backend.model_id)
                all_tags.append(record)

            clips_this_run += len(batch)

            # Progress
            elapsed = time.time() - start_time
            rate = clips_this_run / elapsed if elapsed > 0 else 0
            remaining = clip_limit - clips_this_run
            eta_min = remaining / rate / 60 if rate > 0 else 0
            print(f"  [{clips_this_run:,}/{clip_limit:,}] "
                  f"{rate:.2f} clips/s | ETA {eta_min:.0f} min | {backend.model_name}")
            log_metrics(wb_run, {
                "clips_tagged": clips_this_run,
                "throughput_clips_per_s": rate,
                "eta_min": eta_min,
            })

            # Checkpoint
            if clips_this_run % CHECKPOINT_EVERY < batch_size:
                save_checkpoint(all_tags, tags_file)
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
        save_checkpoint(all_tags, tags_file)
        stop_event.set()
        producer.join(timeout=10)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if clips_this_run > 0:
        elapsed = time.time() - start_time
        print(f"\nSegment done: {clips_this_run:,} clips in {elapsed:.0f}s "
              f"({clips_this_run/elapsed:.2f} clips/s)")

    return all_tags


# ═════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR / WORKER (subprocess pattern for VRAM leak management)
# ═════════════════════════════════════════════════════════════════════════

def orchestrator_main(args):
    """Spawn worker subprocesses every ENGINE_RESTART_EVERY clips."""
    tags_file = get_tags_file(args.model, args.BAKEOFF, args.subset)

    if args.SANITY:
        total_clips = 20
    elif args.BAKEOFF:
        total_clips = BAKEOFF_CLIP_COUNT
    else:
        total_clips = TOTAL_CLIPS

    # For subset+FULL mode: total is subset size minus already-bakeoff'd
    if args.subset and args.FULL:
        subset_keys = load_subset(args.subset)
        total_clips = len(subset_keys)

    all_tags, skip_count = load_checkpoint(tags_file)
    if skip_count >= total_clips:
        print(f"Already complete: {skip_count:,}/{total_clips:,} clips tagged")
        generate_plot(all_tags, args.model, tags_file.parent)
        return

    if skip_count > 0:
        print(f"Resuming from checkpoint: {skip_count:,}/{total_clips:,} clips")

    segment_idx = 0
    while skip_count < total_clips:
        segment_size = min(ENGINE_RESTART_EVERY, total_clips - skip_count)
        segment_idx += 1
        print(f"\n{'='*60}")
        print(f"WORKER {segment_idx} [{args.model}]: clips {skip_count:,} → {skip_count + segment_size:,}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "-u", os.path.abspath(__file__),
            "--_worker",
            "--model", args.model,
            "--start-from", str(skip_count),
            "--process-count", str(segment_size),
            "--batch-size", str(get_batch_size(args.model, args.batch_size)),
        ]
        if args.SANITY:
            cmd.append("--SANITY")
        if args.BAKEOFF:
            cmd.append("--BAKEOFF")
        if args.FULL:
            cmd.append("--FULL")
        if args.subset:
            cmd.extend(["--subset", args.subset])
        if args.no_wandb:
            cmd.append("--no-wandb")

        result = subprocess.run(cmd)

        new_tags, new_count = load_checkpoint(tags_file)
        if new_count > skip_count:
            skip_count = new_count
            all_tags = new_tags
            print(f"Worker done. Progress: {skip_count:,}/{total_clips:,}")
        elif result.returncode != 0:
            print(f"Worker failed (exit {result.returncode}). Resume with same command.")
            break
        else:
            skip_count = total_clips
            all_tags = new_tags

    print(f"\n=== TAGGING COMPLETE ({args.model}) ===")
    print(f"Saved: {tags_file}")
    print(f"Total clips tagged: {len(all_tags):,}")
    generate_plot(all_tags, args.model, tags_file.parent)


def worker_main(args):
    """Worker subprocess: load backend, process segment, exit."""
    check_gpu()

    mode = "SANITY" if args.SANITY else ("BAKEOFF" if args.BAKEOFF else ("POC" if args.subset else "FULL"))
    wb_run = init_wandb("m04", f"{mode}_{args.model}",
                        config={"model": args.model, "start_from": args.start_from,
                                "process_count": args.process_count},
                        enabled=not args.no_wandb)

    tags_file = get_tags_file(args.model, args.BAKEOFF, args.subset)
    subset_keys = load_subset(args.subset) if args.subset else set()

    backend_cls = BACKENDS[args.model]
    backend = backend_cls()
    backend.load_model()
    print(f"Backend loaded: {backend.model_name} | batch_size={args.batch_size}")

    stream_and_tag(
        backend, args,
        start_from=args.start_from,
        clip_limit=args.process_count,
        tags_file=tags_file,
        subset_keys=subset_keys,
        wb_run=wb_run,
    )

    backend.cleanup()

    log_artifact(wb_run, f"tags_{args.model}", str(tags_file))
    finish_wandb(wb_run)
    print("Worker exiting (GPU memory will be released).")


# ═════════════════════════════════════════════════════════════════════════
# DUMMY MODE (no GPU, for testing pipeline)
# ═════════════════════════════════════════════════════════════════════════

def stream_and_tag_dummy(args) -> list:
    """Dummy tagging: streams from HF, assigns default tag values."""
    if not HAS_DATASETS:
        print("ERROR: datasets library required")
        sys.exit(1)

    tags_file = get_tags_file(args.model, args.BAKEOFF, args.subset)
    subset_keys = load_subset(args.subset) if args.subset else set()

    print(f"\nStreaming from: {HF_DATASET_REPO} (dummy mode, model={args.model})")
    ds = load_dataset(HF_DATASET_REPO, split="train", streaming=True)
    ds = ds.decode(False)

    clip_limit = 20 if args.SANITY else (BAKEOFF_CLIP_COUNT if args.BAKEOFF else 100)

    all_tags = []
    for example in ds:
        if subset_keys:
            clip_key = get_clip_key(example)
            if clip_key not in subset_keys:
                continue

        tags = get_dummy_tag()
        record = add_provenance(tags, example, VLM_MODELS.get(args.model, "dummy"))
        all_tags.append(record)

        if len(all_tags) % 10 == 0:
            print(f"  [dummy] {len(all_tags)}/{clip_limit} clips tagged")

        if len(all_tags) >= clip_limit:
            break

    save_checkpoint(all_tags, tags_file)
    return all_tags


# ═════════════════════════════════════════════════════════════════════════
# PLOT
# ═════════════════════════════════════════════════════════════════════════

def generate_plot(all_tags: list, model_name: str, output_dir: Path = None):
    """Generate scene distribution bar chart (.png + .pdf)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plot")
        return

    if output_dir is None:
        output_dir = OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_counts = {}
    for t in all_tags:
        st = t.get("scene_type", "unknown")
        scene_counts[st] = scene_counts.get(st, 0) + 1

    print(f"\nScene type distribution ({model_name}):")
    for scene, count in sorted(scene_counts.items(), key=lambda x: -x[1]):
        print(f"  {scene}: {count}")

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
    ax.set_title(f'{model_name} Scene Type Distribution (n={len(all_tags):,} clips)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plot_png = output_dir / f"m04_scene_distribution_{model_name}.png"
    plot_pdf = output_dir / f"m04_scene_distribution_{model_name}.pdf"
    plt.savefig(plot_png, dpi=150)
    plt.savefig(plot_pdf)
    plt.close()
    print(f"Saved: {plot_png}")
    print(f"Saved: {plot_pdf}")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VLM tagging with bake-off (3 backends, HF WebDataset streaming)")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(BACKENDS.keys()),
                        help="VLM backend: qwen, videollama, keye")
    parser.add_argument("--SANITY", action="store_true", help="Process 20 clips only")
    parser.add_argument("--BAKEOFF", action="store_true",
                        help=f"Bake-off mode: tag first {BAKEOFF_CLIP_COUNT} clips")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    parser.add_argument("--dummy", action="store_true", help="Dummy tags (no GPU)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    add_subset_arg(parser)
    add_wandb_args(parser)

    # Internal worker args (spawned by orchestrator)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start-from", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--process-count", type=int, default=ENGINE_RESTART_EVERY,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Worker mode
    if args._worker:
        worker_main(args)
        return

    if not (args.SANITY or args.BAKEOFF or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --BAKEOFF, or --FULL")
        sys.exit(1)

    tags_file = get_tags_file(args.model, args.BAKEOFF, args.subset)
    print(f"Model:   {args.model} ({VLM_MODELS[args.model]})")
    print(f"Mode:    {'SANITY' if args.SANITY else 'BAKEOFF' if args.BAKEOFF else 'FULL'}")
    print(f"Output:  {tags_file}")
    if args.subset:
        print(f"Subset:  {args.subset}")

    # Check existing
    if tags_file.exists() and not args.dummy:
        total = 20 if args.SANITY else (BAKEOFF_CLIP_COUNT if args.BAKEOFF else TOTAL_CLIPS)
        all_tags, count = load_checkpoint(tags_file)
        if count >= total:
            if not check_output_exists([tags_file], "tags"):
                print(f"Loaded {count:,} cached tags (complete)")
                generate_plot(all_tags, args.model, tags_file.parent)
                return

    # Dummy mode
    if args.dummy:
        all_tags = stream_and_tag_dummy(args)
        print(f"\n=== TAGGING COMPLETE (dummy, {args.model}) ===")
        print(f"Saved: {tags_file}")
        print(f"Total: {len(all_tags):,}")
        generate_plot(all_tags, args.model, tags_file.parent)
    else:
        orchestrator_main(args)


if __name__ == "__main__":
    main()
