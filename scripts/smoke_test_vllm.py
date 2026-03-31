"""
Simplest possible vLLM + Qwen3-VL video inference test.
NOT part of the pipeline. Just verifies vLLM works on your GPU.

SETUP (separate venv — never install vLLM into venv_walkindia):
    uv venv venv_vllm --python 3.12
    source venv_vllm/bin/activate
    uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
    uv pip install qwen-vl-utils transformers

RUN:
    source venv_vllm/bin/activate
    python scripts/smoke_test_vllm.py 2>&1 | tee logs/vllm_smoke.log
"""
import sys
import time

# Step 1: Verify imports
print("=== Step 1: Verify imports ===")
try:
    from vllm import LLM, SamplingParams
    print(f"  vLLM imported OK")
except ImportError as e:
    print(f"  FATAL: vLLM not installed: {e}")
    print("  Install: uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly")
    sys.exit(1)

try:
    from qwen_vl_utils import process_vision_info
    print(f"  qwen_vl_utils imported OK")
except ImportError as e:
    print(f"  FATAL: qwen_vl_utils not installed: {e}")
    print("  Install: uv pip install qwen-vl-utils")
    sys.exit(1)

try:
    from transformers import AutoProcessor
    print(f"  transformers imported OK")
except ImportError as e:
    print(f"  FATAL: transformers not installed: {e}")
    sys.exit(1)

import torch
print(f"  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB")

# Step 2: Create LLM engine
print("\n=== Step 2: Create LLM engine ===")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"

t0 = time.time()
llm = LLM(
    model=MODEL,
    max_model_len=4096,
    max_num_seqs=2,
    limit_mm_per_prompt={"video": 1},
    mm_processor_kwargs={"fps": 1},
    gpu_memory_utilization=0.90,
    enforce_eager=True,  # safer, skip CUDA graphs
    trust_remote_code=True,
)
print(f"  Engine created in {time.time() - t0:.1f}s")

# Step 3: Build a test prompt (image, not video — simpler test)
print("\n=== Step 3: Test with image ===")
processor = AutoProcessor.from_pretrained(MODEL)

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"},
        {"type": "text", "text": "What do you see? Reply in one sentence."},
    ]}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages,
    return_video_kwargs=True,
)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs

llm_input = {
    "prompt": text,
    "multi_modal_data": mm_data,
}

sampling_params = SamplingParams(temperature=0, max_tokens=128)

t0 = time.time()
outputs = llm.generate([llm_input], sampling_params=sampling_params)
elapsed = time.time() - t0

print(f"  Response: {outputs[0].outputs[0].text}")
print(f"  Time: {elapsed:.1f}s")

# Step 4: Test with local video (if available)
print("\n=== Step 4: Test with video ===")
import glob
video_files = glob.glob("data/subset_10k_local/*.tar")
if not video_files:
    video_files = glob.glob("data/val_1k_local/*.tar")

if video_files:
    # Extract first video from first TAR
    import tarfile
    import tempfile
    tar_path = video_files[0]
    tmp_dir = tempfile.mkdtemp()
    video_path = None

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mp4"):
                tar.extract(member, tmp_dir)
                video_path = f"{tmp_dir}/{member.name}"
                break

    if video_path:
        print(f"  Testing with: {video_path}")
        video_messages = [
            {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "Describe this Indian street scene in one sentence."},
            ]}
        ]

        text = processor.apply_chat_template(video_messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in, vid_kw = process_vision_info(
            video_messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data = {}
        if vid_in is not None:
            mm_data["video"] = vid_in

        llm_input = {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": vid_kw,
        }

        t0 = time.time()
        outputs = llm.generate([llm_input], sampling_params=sampling_params)
        elapsed = time.time() - t0
        print(f"  Response: {outputs[0].outputs[0].text}")
        print(f"  Time: {elapsed:.1f}s")

        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print("  No .mp4 found in TAR")
else:
    print("  No local video data found — skipping video test")
    print("  (rsync data/ from Mac to test with real videos)")

print("\n=== SMOKE TEST COMPLETE ===")
print("vLLM + Qwen3-VL works on this GPU.")
print("Next: integrate into src/m04_vlm_tag.py")
