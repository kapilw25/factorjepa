"""
VLM tagging via vLLM (Qwen3-VL). Replaces m04_vlm_tag.py for 115K full run.
vLLM handles continuous batching + PagedAttention — no AdaptiveBatchSizer needed.
Requires separate venv_vllm (NEVER install vLLM into venv_walkindia).

SETUP:
    source venv_vllm/bin/activate
    python scripts/smoke_test_vllm.py  # verify vLLM works first

USAGE:
    source venv_vllm/bin/activate
    python -u src/m04_vlm_tag_vllm.py --SANITY 2>&1 | tee logs/m04_vllm_sanity.log
    python -u src/m04_vlm_tag_vllm.py --POC --subset data/subset_10k.json \
        --local-data data/subset_10k_local 2>&1 | tee logs/m04_vllm_poc.log
    python -u src/m04_vlm_tag_vllm.py --FULL --local-data data/full_local \
        2>&1 | tee logs/m04_vllm_full.log
"""
import argparse
import gc
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    TAGS_FILE, TAG_TAXONOMY_JSON, HF_DATASET_REPO,
    check_gpu, load_subset, add_subset_arg, add_local_data_arg,
    get_pipeline_config, get_sanity_clip_limit, get_total_clips,
)
from utils.data_download import ensure_local_data
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, log_artifact, finish_wandb,
)

# vLLM imports (fail loud if not installed)
try:
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
except ImportError as e:
    print(f"FATAL: {e}")
    print("This script requires venv_vllm. Setup:")
    print("  source venv_vllm/bin/activate")
    print("  uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly")
    sys.exit(1)

# Constants
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
_pcfg = get_pipeline_config()
CHECKPOINT_EVERY = _pcfg["streaming"]["checkpoint_every_vlm"]
CLIPS_PER_VLLM_BATCH = 50
SANITY_CLIP_LIMIT = get_sanity_clip_limit("vlm")


# ═════════════════════════════════════════════════════════════════════════
# TAXONOMY + PROMPT (shared with m04_vlm_tag.py)
# ═════════════════════════════════════════════════════════════════════════

def load_taxonomy():
    with open(TAG_TAXONOMY_JSON, 'r') as f:
        taxonomy = json.load(f)
    return {k: v for k, v in taxonomy.items() if not k.startswith("_")}


TAXONOMY = load_taxonomy()


def build_tag_prompt(taxonomy):
    lines, conf_lines, multi_fields = [], [], []
    for field, spec in taxonomy.items():
        values = spec["values"]
        if spec["type"] == "single":
            lines.append('  "{}": "{}"'.format(field, "|".join(values)))
        else:
            multi_fields.append(field)
            lines.append('  "{}": ["subset of: {}"]'.format(field, ", ".join(values)))
        conf_lines.append('  "confidence_{}": 0.0-1.0'.format(field))

    json_block = "{\n" + ",\n".join(lines) + ",\n" + ",\n".join(conf_lines) + "\n}"
    multi_note = 'For "{}": list ALL that apply.'.format('" and "'.join(multi_fields)) if multi_fields else ""

    return (
        "Analyze this Indian street video clip. Output ONLY a JSON object with these fields:\n\n"
        + json_block + "\n\n"
        + multi_note + "\n"
        + "For all other fields: pick exactly ONE value.\n"
        + "For confidence_* fields: output your confidence as a float in [0.0, 1.0].\n"
        + "Output ONLY the JSON, no explanation."
    )


TAG_PROMPT = build_tag_prompt(TAXONOMY)


def validate_tag_fields(tag):
    for field, spec in TAXONOMY.items():
        if field not in tag:
            tag[field] = spec["default"]
        elif spec["type"] == "single":
            if tag[field] not in spec["values"]:
                tag[field] = "unsure"
        elif spec["type"] == "multi":
            if isinstance(tag[field], list):
                tag[field] = [v for v in tag[field] if v in spec["values"]]
            else:
                tag[field] = []
    return tag


def parse_json_output(output_text):
    """Extract and validate JSON from VLM output. Returns validated dict or None."""
    start = output_text.find('{')
    end = output_text.rfind('}') + 1
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(output_text[start:end])
    except json.JSONDecodeError:
        return None  # counted as dummy tag by caller
    return validate_tag_fields(parsed)


def get_dummy_tag():
    return {field: spec["default"] for field, spec in TAXONOMY.items()}


# ═════════════════════════════════════════════════════════════════════════
# DATA: Iterate clips from local WebDataset TARs
# ═════════════════════════════════════════════════════════════════════════

def get_clip_key(json_bytes):
    try:
        meta = json.loads(json_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    return "{}/{}/{}".format(
        meta.get("section", ""), meta.get("video_id", ""), meta.get("source_file", ""))


def iter_clips_from_local(local_data, subset_keys=None, processed_keys=None):
    """Yield (clip_key, mp4_path, meta_dict) from local WebDataset TARs."""
    tar_files = sorted(Path(local_data).glob("*.tar"))
    tmp_dir = tempfile.mkdtemp(prefix="m04_vllm_")

    try:
        for tar_path in tar_files:
            with tarfile.open(tar_path, "r") as tar:
                entries = {}
                for member in tar.getmembers():
                    base = member.name.rsplit(".", 1)[0]
                    ext = member.name.rsplit(".", 1)[-1] if "." in member.name else ""
                    entries.setdefault(base, {})[ext] = member

                for base, parts in entries.items():
                    if "json" not in parts or "mp4" not in parts:
                        continue
                    json_bytes = tar.extractfile(parts["json"]).read()
                    clip_key = get_clip_key(json_bytes)
                    if subset_keys and clip_key not in subset_keys:
                        continue
                    if processed_keys and clip_key in processed_keys:
                        continue
                    mp4_bytes = tar.extractfile(parts["mp4"]).read()
                    if not mp4_bytes or len(mp4_bytes) < 1000:
                        continue
                    safe_name = base.replace("/", "_")
                    mp4_path = os.path.join(tmp_dir, safe_name + ".mp4")
                    with open(mp4_path, "wb") as f:
                        f.write(mp4_bytes)
                    yield clip_key, mp4_path, json.loads(json_bytes)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════

def save_checkpoint(tags, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(tags, f)
    os.replace(tmp, path)


def load_checkpoint(path):
    if path.exists():
        try:
            with open(path) as f:
                tags = json.load(f)
            print("Checkpoint loaded: {:,} tags".format(len(tags)))
            return tags
        except (json.JSONDecodeError, OSError) as e:
            print("  WARNING: checkpoint corrupt ({}), starting fresh".format(e))
    return []


# ═════════════════════════════════════════════════════════════════════════
# MAIN TAGGING LOOP
# ═════════════════════════════════════════════════════════════════════════

def tag_clips(args):
    check_gpu()

    from utils.config import get_output_dir
    output_dir = get_output_dir(args.subset, sanity=args.SANITY)
    output_dir.mkdir(parents=True, exist_ok=True)
    tags_file = output_dir / "tags.json"
    checkpoint_path = output_dir / ".m04_vllm_checkpoint.json"

    if tags_file.exists() and not checkpoint_path.exists():
        print("Tags already exist: {}".format(tags_file))
        return

    local_data = getattr(args, "local_data", None)
    subset_keys = load_subset(args.subset) if args.subset else set()
    clip_limit = SANITY_CLIP_LIMIT if args.SANITY else (len(subset_keys) if subset_keys else get_total_clips(local_data=local_data))

    mode = "SANITY" if args.SANITY else ("POC" if args.subset else "FULL")
    wb_run = init_wandb("m04_vllm", mode, config=vars(args), enabled=not args.no_wandb)

    print("\n=== m04_vlm_tag_vllm: {} ===".format(MODEL_ID))
    print("Mode: {} | Clips: {:,} | Data: {}".format(mode, clip_limit, local_data))

    all_tags = load_checkpoint(checkpoint_path)
    processed_keys = {t.get("_clip_key", "") for t in all_tags}

    # Create vLLM engine
    print("\nLoading vLLM engine...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_ID, max_model_len=4096, max_num_seqs=8,
        limit_mm_per_prompt={"video": 1}, mm_processor_kwargs={"fps": 1},
        gpu_memory_utilization=0.90, enforce_eager=True, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Engine loaded in {:.0f}s".format(time.time() - t0))

    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    pbar = tqdm(total=clip_limit, initial=len(all_tags), desc="m04_vllm", unit="clip")
    start_time = time.time()
    last_window_time, last_window_count = start_time, len(all_tags)
    n_dummy = 0

    batch_inputs, batch_keys, batch_metas = [], [], []

    for clip_key, mp4_path, meta in iter_clips_from_local(
            local_data, subset_keys=subset_keys, processed_keys=processed_keys):

        if len(all_tags) >= clip_limit:
            break

        messages = [{"role": "user", "content": [
            {"type": "video", "video": mp4_path},
            {"type": "text", "text": TAG_PROMPT},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            img_in, vid_in, vid_kw = process_vision_info(
                messages, return_video_kwargs=True, return_video_metadata=True)
        except (RuntimeError, ValueError) as e:
            print("  WARN: process_vision_info failed for {}: {}".format(clip_key, e))
            continue

        mm_data = {}
        if vid_in is not None:
            mm_data["video"] = vid_in

        batch_inputs.append({
            "prompt": text, "multi_modal_data": mm_data,
            "mm_processor_kwargs": vid_kw if vid_kw else {},
        })
        batch_keys.append(clip_key)
        batch_metas.append(meta)

        try:
            os.unlink(mp4_path)
        except OSError:
            pass  # temp file cleanup, non-critical

        if len(batch_inputs) >= CLIPS_PER_VLLM_BATCH:
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
            for out, key, meta_d in zip(outputs, batch_keys, batch_metas):
                tag = parse_json_output(out.outputs[0].text)
                if tag is None:
                    tag = get_dummy_tag()
                    n_dummy += 1
                tag["_clip_key"] = key
                tag["_video_id"] = meta_d.get("video_id", "")
                tag["_model"] = MODEL_ID
                tag["_backend"] = "vllm"
                all_tags.append(tag)
                processed_keys.add(key)

            pbar.update(len(batch_inputs))
            batch_inputs, batch_keys, batch_metas = [], [], []

            now = time.time()
            if now - last_window_time >= 30:
                rate = (len(all_tags) - last_window_count) / (now - last_window_time)
                pbar.set_postfix_str("{:.1f} clips/s | dummy={}".format(rate, n_dummy))
                log_metrics(wb_run, {"clips": len(all_tags), "rate": rate, "dummy": n_dummy}, step=len(all_tags))
                last_window_time, last_window_count = now, len(all_tags)

            if len(all_tags) % CHECKPOINT_EVERY < CLIPS_PER_VLLM_BATCH:
                save_checkpoint(all_tags, checkpoint_path)

    # Final partial batch
    if batch_inputs:
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        for out, key, meta_d in zip(outputs, batch_keys, batch_metas):
            tag = parse_json_output(out.outputs[0].text)
            if tag is None:
                tag = get_dummy_tag()
                n_dummy += 1
            tag["_clip_key"] = key
            tag["_video_id"] = meta_d.get("video_id", "")
            tag["_model"] = MODEL_ID
            tag["_backend"] = "vllm"
            all_tags.append(tag)
        pbar.update(len(batch_inputs))

    pbar.close()
    elapsed = time.time() - start_time

    dummy_pct = n_dummy / max(len(all_tags), 1) * 100
    if dummy_pct > 5:
        print("FATAL: {:.1f}% dummy tags exceeds 5% threshold.".format(dummy_pct))
        sys.exit(1)

    with open(tags_file, "w") as f:
        json.dump(all_tags, f, indent=2)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("\n=== TAGGING COMPLETE ===")
    print("Model:  {} (vLLM)".format(MODEL_ID))
    print("Clips:  {:,} | Dummy: {:,} ({:.1f}%)".format(len(all_tags), n_dummy, dummy_pct))
    print("Time:   {:.1f} min ({:.1f} clips/s)".format(elapsed / 60, len(all_tags) / max(elapsed, 1)))

    log_metrics(wb_run, {"total": len(all_tags), "dummy": n_dummy, "time_min": elapsed / 60})
    log_artifact(wb_run, "tags", str(tags_file))
    finish_wandb(wb_run)


def main():
    parser = argparse.ArgumentParser(description="VLM tagging via vLLM (Qwen3-VL)")
    parser.add_argument("--SANITY", action="store_true", help="Tag first {} clips".format(SANITY_CLIP_LIMIT))
    parser.add_argument("--POC", action="store_true", help="Tag POC subset")
    parser.add_argument("--FULL", action="store_true", help="Tag all clips")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    ensure_local_data(args)
    tag_clips(args)


if __name__ == "__main__":
    main()
