"""
Simplest possible vLLM + Qwen3-VL video inference test.
NOT part of the pipeline. Just verifies vLLM works on your GPU.

SETUP (separate venv -- never install vLLM into venv_walkindia):
    uv venv venv_vllm --python 3.12
    source venv_vllm/bin/activate
    uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
    uv pip install -r requirements_gpu_vllm.txt

RUN:
    source venv_vllm/bin/activate
    python scripts/smoke_test_vllm.py 2>&1 | tee logs/vllm_smoke.log
"""
import sys
import time


def verify_imports():
    """Step 1: Verify all required packages are importable."""
    print("=== Step 1: Verify imports ===")
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
        print("  vLLM imported OK")
    except ImportError as e:
        print(f"  FATAL: vLLM not installed: {e}")
        print("  Install: uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly")
        sys.exit(1)

    try:
        from qwen_vl_utils import process_vision_info  # noqa: F401
        print("  qwen_vl_utils imported OK")
    except ImportError as e:
        print(f"  FATAL: qwen_vl_utils not installed: {e}")
        print("  Install: uv pip install qwen-vl-utils")
        sys.exit(1)

    try:
        from transformers import AutoProcessor  # noqa: F401
        print("  transformers imported OK")
    except ImportError as e:
        print(f"  FATAL: transformers not installed: {e}")
        sys.exit(1)

    import torch
    print(f"  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")


def create_engine(model_id):
    """Step 2: Create vLLM LLM engine."""
    from vllm import LLM

    print("\n=== Step 2: Create LLM engine ===")
    t0 = time.time()
    llm = LLM(
        model=model_id,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"video": 1},
        mm_processor_kwargs={"fps": 1},
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print(f"  Engine created in {time.time() - t0:.1f}s")
    return llm


def test_image(llm, processor, model_id):
    """Step 3: Test with a local synthetic image (no remote fetch — avoids HTTP 403)."""
    from vllm import SamplingParams
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    import tempfile
    import os

    print("\n=== Step 3: Test with image ===")

    # Generate a local test image (avoids remote URL failures from Wikimedia 403 etc.)
    img = Image.new("RGB", (300, 200), color=(70, 130, 180))
    tmp_img = os.path.join(tempfile.mkdtemp(), "test_image.png")
    img.save(tmp_img)
    print(f"  Using local test image: {tmp_img}")

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": tmp_img},
            {"type": "text", "text": "What do you see? Reply in one sentence."},
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    llm_input = {"prompt": text, "multi_modal_data": mm_data}
    sampling_params = SamplingParams(temperature=0, max_tokens=128)

    t0 = time.time()
    outputs = llm.generate([llm_input], sampling_params=sampling_params)
    elapsed = time.time() - t0

    print(f"  Response: {outputs[0].outputs[0].text}")
    print(f"  Time: {elapsed:.1f}s")

    os.unlink(tmp_img)
    return sampling_params


def test_video(llm, processor, sampling_params):
    """Step 4: Test with a local video from TAR shards (if available)."""
    import glob

    from vllm import SamplingParams
    from qwen_vl_utils import process_vision_info

    print("\n=== Step 4: Test with video ===")
    video_files = glob.glob("data/subset_10k_local/*.tar")
    if not video_files:
        video_files = glob.glob("data/val_1k_local/*.tar")

    if not video_files:
        print("  No local video data found -- skipping video test")
        print("  (rsync data/ from Mac to test with real videos)")
        return

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

    if not video_path:
        print("  No .mp4 found in TAR")
        return

    print(f"  Testing with: {video_path}")
    video_messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": "Describe this Indian street scene in one sentence."},
        ]}
    ]

    text = processor.apply_chat_template(video_messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in, vid_kw = process_vision_info(
        video_messages, return_video_kwargs=True, return_video_metadata=True)

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

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    MODEL = "Qwen/Qwen3-VL-8B-Instruct"

    verify_imports()

    from transformers import AutoProcessor

    llm = create_engine(MODEL)
    processor = AutoProcessor.from_pretrained(MODEL)

    sampling_params = test_image(llm, processor, MODEL)
    test_video(llm, processor, sampling_params)

    print("\n=== SMOKE TEST COMPLETE ===")
    print("vLLM + Qwen3-VL works on this GPU.")


# vLLM v0.18+ uses 'spawn' multiprocessing: child processes re-import this file.
# Without this guard, LLM() runs again in the child → infinite recursion → crash.
if __name__ == "__main__":
    main()
