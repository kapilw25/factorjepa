"""
VLM tagging with bake-off: 3 backends (Qwen3-VL, VideoLLaMA3, LLaVA-NeXT-Video — all via transformers).
Orchestrator/worker pattern for VRAM management. HF WebDataset streaming with checkpoint/resume.

USAGE:
    python -u src/m04_vlm_tag.py --model qwen --SANITY 2>&1 | tee logs/m04_sanity_qwen.log
    python -u src/m04_vlm_tag.py --model qwen --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_qwen_poc.log
    python -u src/m04_vlm_tag.py --model videollama --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_videollama_poc.log
    python -u src/m04_vlm_tag.py --model llava --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_llava_poc.log
    python -u src/m04_vlm_tag.py --model qwen --FULL --subset data/subset_10k.json 2>&1 | tee logs/m04_full_qwen_poc.log
    python -u src/m04_vlm_tag.py --model qwen --plot-only --subset data/subset_10k.json 2>&1 | tee logs/m04_full_qwen_plot.log
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

from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    TAGS_FILE, TAG_TAXONOMY_JSON, HF_DATASET_REPO, OUTPUTS_DIR,
    VLM_MODELS, BAKEOFF_CLIP_COUNT, BAKEOFF_DIR, OUTPUTS_POC_DIR, OUTPUTS_SANITY_DIR,
    check_gpu, check_output_exists, load_subset, add_subset_arg, add_local_data_arg,
)
from utils.gpu_batch import compute_batch_sizes, add_gpu_mem_arg, AdaptiveBatchSizer
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

class VLMBackend(ABC):
    """Abstract base class for VLM tagging backends."""

    PREPROCESS_WORKERS = 4  # Default; subclasses override

    def __init__(self, model_name: str, model_id: str):
        self.model_name = model_name
        self.model_id = model_id
        self.batch_sizer = None  # Set by worker_main() after load_model()

    @abstractmethod
    def load_model(self) -> None:
        """Load model + processor into GPU memory."""

    @abstractmethod
    def preprocess_one(self, example: dict, tmp_dir: str) -> dict | None:
        """Preprocess a single clip. Returns backend-specific dict or None on failure."""

    def preprocess_batch(self, batch: list, tmp_dir: str) -> list:
        """Parallel preprocessing via ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.PREPROCESS_WORKERS) as pool:
            return list(pool.map(lambda ex: self.preprocess_one(ex, tmp_dir), batch))

    def generate_batch(self, preprocessed: list, batch: list) -> list:
        """Orchestrate adaptive sub-batching with OOM recovery.

        Filters None-preprocessed → dummy tags, then processes valid items
        in sub-batches sized by self.batch_sizer. On OOM: halve sub-batch,
        empty_cache, retry. Cleanup happens OUTSIDE except block (fairseq pattern).
        """
        import torch

        results = [None] * len(batch)

        # Fill None-preprocessed slots with dummy tags
        valid = []
        for i, pp in enumerate(preprocessed):
            if pp is None:
                results[i] = get_dummy_tag()
            else:
                valid.append((i, pp))

        if not valid:
            return results

        # No sizer configured → process all valid items in one call (legacy path)
        if self.batch_sizer is None:
            sub_results = self._generate_subbatch([pp for _, pp in valid])
            for (orig_idx, _), tag in zip(valid, sub_results):
                results[orig_idx] = tag
            return results

        # Process valid items in adaptive sub-batches
        idx = 0
        while idx < len(valid):
            sub_size = min(self.batch_sizer.size, len(valid) - idx)
            sub_items = valid[idx:idx + sub_size]
            sub_pp = [pp for _, pp in sub_items]

            oom = False
            sub_results = None
            try:
                sub_results = self._generate_subbatch(sub_pp)
            except torch.cuda.OutOfMemoryError:
                oom = True

            # CRITICAL: cleanup OUTSIDE except block (exception holds stack frame
            # references → prevents tensor deallocation if cleaned inside except)
            if oom:
                gc.collect()
                torch.cuda.empty_cache()
                if not self.batch_sizer.on_oom():
                    # At min size, still OOM → dummy-tag this sub-batch, move on
                    for orig_idx, _ in sub_items:
                        results[orig_idx] = get_dummy_tag()
                    self._cleanup_preprocessed(sub_pp)
                    idx += sub_size
                continue  # retry same sub_items with reduced size (idx not advanced)

            # Success — map results back
            for (orig_idx, _), tag in zip(sub_items, sub_results):
                results[orig_idx] = tag
            idx += sub_size
            self.batch_sizer.after_batch_success()

        return results

    @abstractmethod
    def _generate_subbatch(self, items: list[dict]) -> list[dict]:
        """Run inference on a sub-batch of preprocessed items (all non-None).

        Args:
            items: List of preprocessed dicts from preprocess_one() (guaranteed non-None).
        Returns:
            List of tag dicts, same length as items.
        """

    def _cleanup_preprocessed(self, items: list[dict]) -> None:
        """Optional hook: clean up preprocessed resources when items are dummy-tagged.

        Called by generate_batch() when OOM at min sub-batch forces dummy-tagging.
        Override in backends that create temp files (e.g., VideoLLaMA3).
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release GPU memory."""


# ═════════════════════════════════════════════════════════════════════════
# BACKEND: Qwen3-VL-8B (transformers, batched inference)
# ═════════════════════════════════════════════════════════════════════════

class QwenBackend(VLMBackend):
    """Qwen3-VL-8B via transformers. Batched model.generate() with adaptive sub-batching."""

    PREPROCESS_WORKERS = 6  # Heavy CPU work (decord decode via process_vision_info) in preprocess_one

    def __init__(self):
        super().__init__("qwen", VLM_MODELS["qwen"])
        self.model = None
        self.processor = None

    def load_model(self):
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        print(f"Loading transformers: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        print(f"Qwen loaded via transformers (FA2)")

    def preprocess_one(self, example: dict, tmp_dir: str) -> dict | None:
        """Write mp4 → decode video (decord, fps=1) → template text. Runs in ThreadPoolExecutor."""
        tmp_path = None
        try:
            from qwen_vl_utils import process_vision_info

            mp4_data = example["mp4"]
            mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
            key = example.get("__key__", "unknown")

            tmp_path = os.path.join(tmp_dir, f"{key}.mp4")
            with open(tmp_path, "wb") as f:
                f.write(mp4_bytes)

            if not validate_mp4(tmp_path):
                return None

            # CPU-heavy: video decode + frame sampling (thread-safe)
            messages = [{"role": "user", "content": [
                {"type": "video", "video": tmp_path, "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": TAG_PROMPT},
            ]}]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            _image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )

            return {
                "text": text,
                "video_inputs": video_inputs,
                "video_kwargs": video_kwargs,
                "key": key,
            }
        except Exception as e:
            print(f"  WARN: qwen preprocess failed ({example.get('__key__', '?')}): {e}")
            return None
        finally:
            # Delete temp file immediately — video already decoded into tensors
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _generate_subbatch(self, items: list[dict]) -> list[dict]:
        """Batched model.generate(). OOM propagates to base class AdaptiveBatchSizer."""
        import torch

        try:
            return self._generate_batched(items)
        except torch.cuda.OutOfMemoryError:
            raise  # → base class handles via AdaptiveBatchSizer
        except Exception as e:
            print(f"  WARN: qwen batched inference failed ({e}), per-clip fallback")
            return self._generate_per_clip(items)

    def _generate_batched(self, items: list[dict]) -> list[dict]:
        """True batched path: single processor() + single model.generate()."""
        import torch

        # Collect texts and video inputs across the sub-batch
        texts = [pp["text"] for pp in items]
        all_videos = []
        all_fps = []
        for pp in items:
            all_videos.extend(pp["video_inputs"])
            all_fps.extend(pp["video_kwargs"].get("fps", []))

        merged_kwargs = {}
        if all_fps:
            merged_kwargs = {"do_sample_frames": False, "fps": all_fps}

        # Batched tokenization + vision processing (left-pad for decoder-only generation)
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(
            text=texts,
            videos=all_videos,
            return_tensors="pt",
            padding=True,
            **merged_kwargs,
        ).to(self.model.device)

        # Single batched generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Strip input tokens (left-padded → all input_ids same length)
        seq_len = inputs["input_ids"].shape[1]
        responses = self.processor.batch_decode(
            output_ids[:, seq_len:], skip_special_tokens=True
        )

        results = []
        for resp in responses:
            tags = parse_json_output(resp)
            results.append(tags if tags is not None else get_dummy_tag())
        return results

    def _generate_per_clip(self, items: list[dict]) -> list[dict]:
        """Per-clip fallback using already-decoded video tensors from preprocess_one."""
        import torch

        results = []
        for pp in items:
            try:
                inputs = self.processor(
                    text=[pp["text"]],
                    videos=pp["video_inputs"],
                    return_tensors="pt",
                    padding=True,
                    **pp["video_kwargs"],
                ).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=512, temperature=0.1, do_sample=True
                    )
                generated = output_ids[0][inputs["input_ids"].shape[1]:]
                response = self.processor.decode(generated, skip_special_tokens=True)

                tags = parse_json_output(response)
                results.append(tags if tags is not None else get_dummy_tag())
            except Exception as e:
                print(f"  WARN: qwen per-clip failed ({pp['key']}): {e}")
                results.append(get_dummy_tag())
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
# BACKEND: VideoLLaMA3-7B (transformers, batched inference)
# ═════════════════════════════════════════════════════════════════════════

class VideoLLaMA3Backend(VLMBackend):
    """VideoLLaMA3-7B via transformers. Per-clip inference (token compression enforces batch_size=1).
    Processor is NOT thread-safe (trust_remote_code) — kept in GPU thread."""

    def __init__(self):
        super().__init__("videollama", VLM_MODELS["videollama"])
        self.model = None
        self.processor = None

    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        # VideoLLaMA3's remote code imports VideoInput from transformers.image_utils,
        # but VideoInput is only a type hint and doesn't exist in transformers <5.0.
        import transformers.image_utils as _img_utils
        if not hasattr(_img_utils, "VideoInput"):
            import typing as _t
            _img_utils.VideoInput = _t.Any

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
        """Write mp4 + validate. Processor call stays in GPU thread (not thread-safe)."""
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

    def _generate_subbatch(self, items: list[dict]) -> list[dict]:
        """Per-clip sequential: processor + model.generate() in GPU thread."""
        import torch

        results = []
        for pp in items:
            video_path = pp["video_path"]
            try:
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 128}},
                        {"type": "text", "text": TAG_PROMPT},
                    ]},
                ]

                inputs = self.processor(conversation=conversation, return_tensors="pt")
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=512, temperature=0.1, do_sample=True
                    )
                response = self.processor.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()

                tags = parse_json_output(response)
                results.append(tags if tags is not None else get_dummy_tag())
            except Exception as e:
                print(f"  WARN: videollama per-clip failed ({pp['key']}): {e}")
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
# BACKEND: LLaVA-NeXT-Video-7B (native transformers, batched inference)
# ═════════════════════════════════════════════════════════════════════════

LLAVA_NUM_FRAMES = 16  # 8-32 range; 16 balances detail vs VRAM on 24GB


class LLaVANextBackend(VLMBackend):
    """LLaVA-NeXT-Video-7B via native transformers. Batched model.generate()."""

    PREPROCESS_WORKERS = 4  # Already does heavy work (PyAV frame decode)

    def __init__(self):
        super().__init__("llava", VLM_MODELS["llava"])
        self.model = None
        self.processor = None

    def load_model(self):
        import torch
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

        print(f"Loading transformers: {self.model_id}")
        self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_id)
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        print(f"LLaVA-NeXT-Video loaded via transformers (FA2)")

    def preprocess_one(self, example: dict, tmp_dir: str) -> dict | None:
        """PyAV frame decode (CPU-heavy). Runs in ThreadPoolExecutor. No temp files."""
        try:
            import av
            import io
            import numpy as np

            mp4_data = example["mp4"]
            mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
            key = example.get("__key__", "unknown")

            container = av.open(io.BytesIO(mp4_bytes))
            total_frames = container.streams.video[0].frames
            if total_frames <= 0:
                # Fallback: count frames manually
                total_frames = sum(1 for _ in container.decode(video=0))
                container.seek(0)

            n = min(LLAVA_NUM_FRAMES, total_frames)
            indices = set(np.linspace(0, total_frames - 1, n).astype(int))

            frames = []
            container.seek(0)
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    frames.append(frame.to_ndarray(format="rgb24"))
                if i >= max(indices):
                    break
            container.close()

            if len(frames) == 0:
                return None

            clip = np.stack(frames)  # (N, H, W, 3)

            # Pad to LLAVA_NUM_FRAMES for uniform batching (repeat last frame)
            if clip.shape[0] < LLAVA_NUM_FRAMES:
                pad = np.repeat(clip[-1:], LLAVA_NUM_FRAMES - clip.shape[0], axis=0)
                clip = np.concatenate([clip, pad], axis=0)

            return {"clip": clip, "key": key}

        except Exception as e:
            print(f"  WARN: llava preprocess failed ({example.get('__key__', '?')}): {e}")
            return None

    def _generate_subbatch(self, items: list[dict]) -> list[dict]:
        """Batched model.generate(). OOM propagates to base class AdaptiveBatchSizer."""
        import torch

        try:
            return self._generate_batched(items)
        except torch.cuda.OutOfMemoryError:
            raise  # → base class handles via AdaptiveBatchSizer
        except Exception as e:
            print(f"  WARN: llava batched inference failed ({e}), per-clip fallback")
            return self._generate_per_clip(items)

    def _generate_batched(self, items: list[dict]) -> list[dict]:
        """True batched path: processor natively batches text+videos → single model.generate()."""
        import torch

        # Same prompt for all clips (tag prompt is fixed)
        conversation = [{"role": "user", "content": [
            {"type": "text", "text": TAG_PROMPT},
            {"type": "video"},
        ]}]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # Batched processor call — handles tokenization, video processing, left-padding
        # pixel_values_videos is stacked (5D) because LLAVA_NUM_FRAMES=16 is fixed
        prompts = [prompt] * len(items)
        clips = [pp["clip"] for pp in items]

        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(
            text=prompts, videos=clips,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        # Single batched generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Strip input tokens (left-padded → all same length)
        seq_len = inputs["input_ids"].shape[1]
        responses = self.processor.batch_decode(
            output_ids[:, seq_len:], skip_special_tokens=True
        )

        results = []
        for resp in responses:
            tags = parse_json_output(resp)
            results.append(tags if tags is not None else get_dummy_tag())
        return results

    def _generate_per_clip(self, items: list[dict]) -> list[dict]:
        """Per-clip fallback using already-decoded numpy clips from preprocess_one."""
        import torch

        results = []
        for pp in items:
            try:
                conversation = [{"role": "user", "content": [
                    {"type": "text", "text": TAG_PROMPT},
                    {"type": "video"},
                ]}]

                prompt = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=prompt, videos=pp["clip"],
                    padding=True, return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=512, do_sample=False
                    )
                generated = output_ids[0][inputs["input_ids"].shape[1]:]
                response = self.processor.decode(generated, skip_special_tokens=True)

                tags = parse_json_output(response)
                results.append(tags if tags is not None else get_dummy_tag())
            except Exception as e:
                print(f"  WARN: llava per-clip failed ({pp['key']}): {e}")
                results.append(get_dummy_tag())

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
    "llava": LLaVANextBackend,
}


# ═════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════

# Default (overridden by auto-compute from gpu_batch.compute_batch_sizes)
TRANSFORMERS_BATCH_SIZE = 4
CHECKPOINT_EVERY = 500
ENGINE_RESTART_EVERY = 10_000
MAX_STREAM_RETRIES = 5
PREFETCH_QUEUE_SIZE = 4  # Batched generate is faster → producer needs more buffer
TOTAL_CLIPS = 115_687
PROMPT_VERSION = "v1.0"


def get_batch_size(model_name: str, override: int = None) -> int:
    if override:
        return override
    return TRANSFORMERS_BATCH_SIZE


# ═════════════════════════════════════════════════════════════════════════
# OUTPUT PATH LOGIC
# ═════════════════════════════════════════════════════════════════════════

def get_tags_file(model_name: str, is_bakeoff: bool, subset_path: str = None,
                   is_sanity: bool = False) -> Path:
    """Determine output tags file based on mode."""
    if is_sanity:
        return OUTPUTS_SANITY_DIR / f"tags_sanity_{model_name}.json"
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

def add_provenance(tags: dict, example: dict, model_id: str,
                    timestamp: float | None = None) -> dict:
    """Merge metadata + tags + provenance into final clip record."""
    meta = example.get("json", {})
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta) if meta else {}
    key = example.get("__key__", "")

    if timestamp is not None:
        ts_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    return {
        "__key__": key,
        **meta,
        **tags,
        "_model": model_id,
        "_prompt_version": PROMPT_VERSION,
        "_tagged_at": ts_str,
    }


# ═════════════════════════════════════════════════════════════════════════
# PIPELINE: Producer / Consumer (backend-agnostic)
# ═════════════════════════════════════════════════════════════════════════

def _create_stream(skip_count: int, local_data: str = None):
    """Create streaming dataset from HF or local WebDataset shards."""
    if local_data:
        ds = load_dataset("webdataset", data_files=f"{local_data}/*.tar", split="train", streaming=True)
    else:
        ds = load_dataset(HF_DATASET_REPO, split="train", streaming=True)
    ds = ds.decode(False)
    if skip_count > 0:
        ds = ds.skip(skip_count)
    return ds


def _producer_thread(backend, start_from, batch_size, tmp_dir,
                     q, stop_event, clip_limit, subset_keys,
                     already_tagged_keys=None, local_data=None):
    """
    Background thread: streams from HF (or local shards), filters by subset,
    skips already-tagged clips on resume, preprocesses batches.
    Puts (batch, preprocessed) onto queue for consumer.
    """
    produced = 0
    retries = 0
    skipped = 0

    while produced < clip_limit and not stop_event.is_set():
        try:
            ds = _create_stream(start_from + produced + skipped, local_data=local_data)
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

                # Dedup on resume — skip already-tagged clips
                if already_tagged_keys:
                    ck = get_clip_key(example)
                    if ck in already_tagged_keys:
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

    # Build dedup set from existing tags for resume correctness (Bug 2 fix)
    already_tagged_keys = None
    if all_tags:
        already_tagged_keys = {t["__key__"] for t in all_tags if "__key__" in t}
        if already_tagged_keys:
            print(f"  [resume] {len(already_tagged_keys):,} already-tagged keys loaded for dedup")

    tmp_base = tags_file.parent / "tmp_m04"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)

    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    batch_size = get_batch_size(backend.model_name, args.batch_size)

    producer = threading.Thread(
        target=_producer_thread,
        args=(backend, start_from, batch_size, tmp_dir,
              q, stop_event, clip_limit, subset_keys,
              already_tagged_keys, getattr(args, 'local_data', None)),
        daemon=True,
    )
    producer.start()

    clips_this_run = 0
    start_time = time.time()
    last_window_time = start_time
    last_window_count = 0
    pbar = tqdm(total=clip_limit, desc=f"m04_{backend.model_name}", unit="clip")

    try:
        while True:
            msg_type, batch, preprocessed = q.get(timeout=600)
            if msg_type == "done":
                break

            batch_t0 = time.time()
            tag_list = backend.generate_batch(preprocessed, batch)
            batch_t1 = time.time()

            # Interpolate _tagged_at across batch duration (real inference time)
            batch_dur = batch_t1 - batch_t0
            for idx, (example, tags) in enumerate(zip(batch, tag_list)):
                frac = idx / max(len(batch) - 1, 1)
                clip_time = batch_t0 + frac * batch_dur
                record = add_provenance(tags, example, backend.model_id, clip_time)
                all_tags.append(record)

            clips_this_run += len(batch)
            pbar.update(len(batch))

            # Windowed throughput
            now = time.time()
            window_elapsed = now - last_window_time
            window_clips = clips_this_run - last_window_count
            if window_elapsed > 0:
                rate = window_clips / window_elapsed
            else:
                rate = 0
            remaining = clip_limit - clips_this_run
            eta_min = remaining / rate / 60 if rate > 0 else 0
            pbar.set_postfix_str(
                f"{rate:.2f} clips/s | ETA {eta_min:.0f} min | {backend.model_name}")
            log_metrics(wb_run, {
                "clips_tagged": clips_this_run,
                "throughput_clips_per_s": rate,
                "eta_min": eta_min,
            })
            # Reset window every 30s
            if window_elapsed >= 30:
                last_window_time = now
                last_window_count = clips_this_run

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
        pbar.close()
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
    tags_file = get_tags_file(args.model, args.BAKEOFF, args.subset, is_sanity=args.SANITY)

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
        ]
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
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
        if args.gpu_mem is not None:
            cmd.extend(["--gpu-mem", str(args.gpu_mem)])
        if getattr(args, 'local_data', None):
            cmd.extend(["--local-data", args.local_data])

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

    # Auto-compute batch size from VRAM if not explicitly set via --batch-size
    global TRANSFORMERS_BATCH_SIZE
    batch_sizes = compute_batch_sizes(gpu_vram_gb=args.gpu_mem)
    if args.batch_size is None:
        TRANSFORMERS_BATCH_SIZE = batch_sizes["transformers"]
        args.batch_size = get_batch_size(args.model)

    mode = "SANITY" if args.SANITY else ("BAKEOFF" if args.BAKEOFF else ("POC" if args.subset else "FULL"))
    wb_run = init_wandb("m04", f"{mode}_{args.model}",
                        config={"model": args.model, "start_from": args.start_from,
                                "process_count": args.process_count},
                        enabled=not args.no_wandb)

    tags_file = get_tags_file(args.model, args.BAKEOFF, args.subset, is_sanity=args.SANITY)
    subset_keys = load_subset(args.subset) if args.subset else set()

    backend_cls = BACKENDS[args.model]
    backend = backend_cls()
    backend.load_model()

    # Adaptive sub-batch sizing: initial from VRAM scaling, max = producer batch size
    sub_batch_size = batch_sizes["transformers_batch"]
    backend.batch_sizer = AdaptiveBatchSizer(
        initial_size=sub_batch_size,
        max_size=args.batch_size,
    )
    print(f"Backend loaded: {backend.model_name} | batch_size={args.batch_size} | {backend.batch_sizer}")

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

    tags_file = get_tags_file(args.model, args.BAKEOFF, args.subset, is_sanity=args.SANITY)
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
    """Generate scene distribution + full taxonomy distribution plots (.png + .pdf)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plot")
        return
    from collections import Counter

    if output_dir is None:
        output_dir = OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Scene type bar chart (single key, backward compat) ──
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

    for ext in ('png', 'pdf'):
        path = output_dir / f"m04_scene_distribution_{model_name}.{ext}"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()

    # ── 2. Full taxonomy distribution (all 11 keys) ──
    tag_keys = [k for k in TAXONOMY]
    n_keys = len(tag_keys)
    cols = 4
    rows = (n_keys + cols) // cols  # extra slot for info panel
    fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows))
    axes = axes.flatten()
    palette = plt.cm.tab20.colors + plt.cm.tab20b.colors  # 40 distinct colors

    for idx, key in enumerate(tag_keys):
        ax = axes[idx]
        meta = TAXONOMY[key]
        tax_values = meta.get("values", [])

        counts = Counter()
        for clip in all_tags:
            val = clip.get(key)
            if val is None:
                counts["MISSING"] += 1
            elif isinstance(val, list):
                for v in val:
                    counts[v] += 1
            else:
                counts[str(val)] += 1

        # Order: taxonomy values first, then off-taxonomy by count
        ordered = []
        remaining = dict(counts)
        for v in tax_values:
            if v in remaining:
                ordered.append((v, remaining.pop(v)))
        for v, c in sorted(remaining.items(), key=lambda x: -x[1]):
            ordered.append((v, c))

        labels = [x[0] for x in ordered]
        values = [x[1] for x in ordered]
        bar_colors = [palette[i % len(palette)] for i in range(len(labels))]

        bars = ax.bar(range(len(labels)), values, color=bar_colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{key} ({meta["type"]})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')

        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(values) * 0.01,
                        str(val), ha='center', va='bottom', fontsize=7)

    # Hide unused subplots, use last for info
    for i in range(n_keys, len(axes)):
        axes[i].axis('off')
    axes[n_keys].text(0.5, 0.5,
                      f'WalkIndia-200K\n{model_name}\n{len(all_tags):,} clips',
                      ha='center', va='center', fontsize=14,
                      transform=axes[n_keys].transAxes)

    fig.suptitle(f'Full Taxonomy Distribution — {model_name} (n={len(all_tags):,} clips)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ('png', 'pdf'):
        path = output_dir / f"m04_taxonomy_distribution_{model_name}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VLM tagging with bake-off (3 backends, HF WebDataset streaming)")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(BACKENDS.keys()),
                        help="VLM backend: qwen, videollama, llava")
    parser.add_argument("--SANITY", action="store_true", help="Process 20 clips only")
    parser.add_argument("--BAKEOFF", action="store_true",
                        help=f"Bake-off mode: tag first {BAKEOFF_CLIP_COUNT} clips")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    parser.add_argument("--dummy", action="store_true", help="Dummy tags (no GPU)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate plots from existing tags (no GPU)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    add_gpu_mem_arg(parser)

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

    if not (args.SANITY or args.BAKEOFF or args.FULL or args.plot_only):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --BAKEOFF, --FULL, or --plot-only")
        sys.exit(1)

    tags_file = get_tags_file(args.model, args.BAKEOFF, args.subset, is_sanity=args.SANITY)
    print(f"Model:   {args.model} ({VLM_MODELS[args.model]})")
    print(f"Output:  {tags_file}")

    # --plot-only: regenerate plots from existing tags, no GPU needed
    if args.plot_only:
        all_tags, count = load_checkpoint(tags_file)
        if count == 0:
            print(f"ERROR: No tags found at {tags_file}")
            sys.exit(1)
        print(f"Loaded {count:,} tags, regenerating plots...")
        generate_plot(all_tags, args.model, tags_file.parent)
        return

    print(f"Mode:    {'SANITY' if args.SANITY else 'BAKEOFF' if args.BAKEOFF else 'FULL'}")
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
