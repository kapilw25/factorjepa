Code-Level Implementation Plan

📁 File Structure

src/
├── m01_scene_understanding.py    # Qwen3-VL-32B object detection
├── m02_instruction_generator.py  # Llama-3.1-70B instruction gen
├── m03_evaluator.py              # VLM-as-Judge (GPT-4o + Prometheus)
├── m04_pipeline_orchestrator.py  # Sequential loading + full pipeline
└── utils/
    ├── __init__.py
    ├── io_handler.py             # JSON/CSV file operations
    └── image_utils.py            # resize, encode, pair-finding

outputs/                              # Created dynamically by each module
├── m01_scenes/                   # Created by m01_scene_understanding.py
│   ├── scene_001.json
│   └── ...
├── m02_instructions/             # Created by m02_instruction_generator.py
│   ├── scene_001.json
│   └── ...
├── m03_evaluations/              # Created by m03_evaluator.py
│   ├── scene_001.json
│   └── ...
└── metrics.csv                   # Created by m03_evaluator.py

---
📄 Module 1: src/m01_scene_understanding.py

"""
Object detection from scene images using Qwen3-VL-32B (8-bit).

Commands:
    python -u src/m01_scene_understanding.py --sanity 2>&1 | tee logs/m01_sanity.log
    python -u src/m01_scene_understanding.py --full --input_dir data/images 2>&1 | tee logs/m01_full.log
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
import json
from pathlib import Path
from utils.io_handler import IOHandler
from utils.image_utils import find_image_pairs, resize_image

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
IMAGE_SIZE = (1024, 1024)
CONFIDENCE_THRESHOLD = 0.7

# ─────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────
class SceneUnderstanding:
    """Qwen3-VL-32B for object detection + scene description."""

    def __init__(self, model_name=MODEL_NAME, image_size=IMAGE_SIZE):
        assert torch.cuda.is_available(), "GPU required"
        self.device = "cuda"

        # 8-bit quantization for A100-80GB
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.image_size = image_size

    def analyze_scene(self, isometric_path: str, topdown_path: str) -> dict:
        """
        Input: 2 images (_isometric.png, _topdown.png)
        Output: JSON with objects + confidence + scene_type
        """
        img_iso = resize_image(isometric_path, self.image_size)
        img_top = resize_image(topdown_path, self.image_size)

        prompt = '''Analyze these two views of the SAME scene (isometric + top-down).
Output ONLY valid JSON with this exact structure:
{
  "objects": [
    {"name": "object_name", "confidence": 0.95},
    ...
  ],
  "scene_type": "outdoor_mall|office|retail|other",
  "layout_description": "brief navigation-relevant description"
}'''

        # Process with Qwen3-VL
        inputs = self.processor(
            text=prompt,
            images=[img_iso, img_top],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=1024)

        response = self.processor.decode(output[0], skip_special_tokens=True)
        return self._parse_json_response(response)

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from model response."""
        # Find JSON block in response
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group())
        return {"objects": [], "scene_type": "unknown", "layout_description": ""}

    def filter_objects(self, objects: list, threshold=CONFIDENCE_THRESHOLD) -> list:
        """Filter low-confidence + deduplicate."""
        seen = set()
        filtered = []
        for obj in objects:
            name = obj["name"].lower().strip()
            if obj["confidence"] >= threshold and name not in seen:
                seen.add(name)
                filtered.append(obj)
        return filtered

    def unload(self):
        """Free GPU memory for next model."""
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        import gc; gc.collect()

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="Run on 1 sample")
    parser.add_argument("--full", action="store_true", help="Run on all images")
    parser.add_argument("--input_dir", type=str, default="data/images")
    args = parser.parse_args()

    # Implementation...

---
📄 Module 2: src/m02_instruction_generator.py

"""
Robot navigation instruction generation using Llama-3.1-70B-Instruct (8-bit).

Commands:
    python -u src/m02_instruction_generator.py --sanity 2>&1 | tee logs/m02_sanity.log
    python -u src/m02_instruction_generator.py --full 2>&1 | tee logs/m02_full.log
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.io_handler import IOHandler

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
NUM_PER_LEVEL = 3
BANNED_TERMS = {"left", "right", "front", "back", "center", "middle"}

# ─────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────
class InstructionGenerator:
    """Llama-3.1-70B for generating robot navigation tasks."""

    def __init__(self, model_name=MODEL_NAME):
        assert torch.cuda.is_available(), "GPU required"

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    def generate_instructions(self, scene_data: dict, num_per_level=NUM_PER_LEVEL) -> dict:
        """
        Input: scene_data from m01 (objects, scene_type, layout)
        Output: {level_1: [...], level_2: [...], level_3: [...]}
        """
        objects = [obj["name"] for obj in scene_data["objects"]]
        objects_str = ", ".join(objects) if objects else "unknown"

        system_prompt = f"""You are a robot task instruction generator.

VALID_OBJECTS: {objects_str}

RULES:
- Use ONLY object names from VALID_OBJECTS for START/END locations
- Format: "Near <object>" for all locations
- NEVER use: left, right, front, back, center, middle
- LEVEL_1: navigation only, INTERACT=none
- LEVEL_2: 1-2 object interactions
- LEVEL_3: 3+ object interactions

OUTPUT FORMAT (one per line):
LEVEL_X | TASK: ... | START: Near <obj> | END: Near <obj> | INTERACT: obj1, obj2
"""

        user_prompt = f"""Scene: {scene_data['scene_type']}
Layout: {scene_data['layout_description']}

Generate {num_per_level} tasks for EACH level (total {num_per_level * 3})."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=2048, temperature=0.6)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_instructions(response, objects)

    def _parse_instructions(self, text: str, valid_objects: list) -> dict:
        """Parse and validate generated instructions."""
        result = {"level_1": [], "level_2": [], "level_3": []}
        # Parsing logic similar to original...
        return result

    def unload(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        import gc; gc.collect()

---
📄 Module 3: src/m03_evaluator.py

"""
VLM-as-Judge evaluation using GPT-4o (baseline) and Prometheus-Vision-13B (scale).

Commands:
    python -u src/m03_evaluator.py --baseline --n_samples 100 2>&1 | tee logs/m03_baseline.log
    python -u src/m03_evaluator.py --scale --use_prometheus 2>&1 | tee logs/m03_scale.log
    python -u src/m03_evaluator.py --metadata_only 2>&1 | tee logs/m03_metadata.log
"""

import os
import base64
import json
from openai import OpenAI
from utils.io_handler import IOHandler
from utils.image_utils import encode_image_base64

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o"
PROMETHEUS_MODEL = "prometheus-eval/prometheus-vision-13b-v1.0"

EVALUATION_RUBRIC = """Rate each criterion from 1-5:
1. Object Accuracy: Are mentioned objects visible in images?
2. Spatial Coherence: Does the navigation path make sense?
3. Task Clarity: Is the instruction unambiguous?
4. Difficulty Alignment: Does complexity match level (1/2/3)?
5. Executability: Could a robot actually perform this?

Output JSON: {"object_accuracy": X, "spatial_coherence": X, "task_clarity": X,
              "difficulty_alignment": X, "executability": X, "reasoning": "..."}"""

# ─────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────
class InstructionEvaluator:
    """Evaluate instruction quality using VLM-as-Judge."""

    def __init__(self, use_openai=True):
        if use_openai:
            self.client = OpenAI()  # Uses OPENAI_API_KEY env var
            self.model = OPENAI_MODEL
        else:
            # Load Prometheus-Vision-13B locally
            self._load_prometheus()

    def evaluate_instruction(
        self,
        isometric_path: str,
        topdown_path: str,
        instruction: str,
        level: int
    ) -> dict:
        """
        Input: 2 images + 1 instruction + level
        Output: scores dict
        """
        img_iso_b64 = encode_image_base64(isometric_path)
        img_top_b64 = encode_image_base64(topdown_path)

        prompt = f"""You are evaluating a robot navigation instruction.

INSTRUCTION (Level {level}):
{instruction}

{EVALUATION_RUBRIC}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_iso_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_top_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=500
        )

        return self._parse_scores(response.choices[0].message.content)

    def evaluate_scene(self, scene_id: str, io: IOHandler) -> dict:
        """Evaluate all instructions for a scene."""
        # Get scene and instructions from JSON files
        scene_data = io.load_scene(scene_id)
        instructions = io.load_instructions(scene_id)

        results = {
            "scene_id": scene_id,
            "level_1": [], "level_2": [], "level_3": []
        }

        for instr in instructions:
            scores = self.evaluate_instruction(
                scene_data["isometric_path"],
                scene_data["topdown_path"],
                instr["text"],
                instr["level"]
            )
            results[f"level_{instr['level']}"].append(scores)

        # Compute averages
        for level in [1, 2, 3]:
            level_scores = results[f"level_{level}"]
            if level_scores:
                results[f"level_{level}_avg"] = {
                    k: sum(s[k] for s in level_scores) / len(level_scores)
                    for k in ["object_accuracy", "spatial_coherence", "task_clarity",
                              "difficulty_alignment", "executability"]
                }

        return results

    def _parse_scores(self, text: str) -> dict:
        """Extract JSON scores from response."""
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group())
        return {}

---
📄 Module 4: src/m04_pipeline_orchestrator.py

"""
Full pipeline orchestrator with sequential model loading for A100-80GB.

Commands:
    python -u src/m04_pipeline_orchestrator.py --sanity 2>&1 | tee logs/m04_sanity.log
    python -u src/m04_pipeline_orchestrator.py --full --input_dir data/images 2>&1 | tee logs/m04_full.log
    python -u src/m04_pipeline_orchestrator.py --eval_only 2>&1 | tee logs/m04_eval.log
"""

import torch
from pathlib import Path
from utils.io_handler import IOHandler
from utils.image_utils import find_image_pairs

# ─────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────
class PipelineOrchestrator:
    """Sequential loading: VLM → unload → LLM → unload → Eval"""

    def __init__(self, output_dir="outputs"):
        self.io = IOHandler(output_dir)

    def run_full_pipeline(self, input_dir: str):
        """
        Step 1: Load Qwen3-VL-32B → detect objects → unload
        Step 2: Load Llama-3.1-70B → generate instructions → unload
        Step 3: Evaluate with GPT-4o API (no local GPU needed)
        """
        pairs = find_image_pairs(input_dir)
        print(f"Found {len(pairs)} image pairs")

        # ─────────────────────────────────────────────────────────
        # STEP 1: Object Detection (GPU: ~40GB)
        # ─────────────────────────────────────────────────────────
        print("\n[STEP 1] Loading Qwen3-VL-32B for object detection...")
        from m01_scene_understanding import SceneUnderstanding

        scene_analyzer = SceneUnderstanding()

        for scene_id, paths in pairs.items():
            print(f"  Processing: {scene_id}")
            scene_data = scene_analyzer.analyze_scene(
                paths["isometric"],
                paths["topdown"]
            )
            scene_data["objects"] = scene_analyzer.filter_objects(scene_data["objects"])

            # Save to JSON
            self.io.save_scene(scene_id, paths, scene_data)

        # FREE GPU MEMORY
        scene_analyzer.unload()
        print("  ✅ VLM unloaded, GPU memory freed")

        # ─────────────────────────────────────────────────────────
        # STEP 2: Instruction Generation (GPU: ~70GB)
        # ─────────────────────────────────────────────────────────
        print("\n[STEP 2] Loading Llama-3.1-70B for instruction generation...")
        from m02_instruction_generator import InstructionGenerator

        instr_gen = InstructionGenerator()

        for scene_id in pairs.keys():
            scene_data = self.io.load_scene(scene_id)
            instructions = instr_gen.generate_instructions(scene_data)

            # Save to JSON
            self.io.save_instructions(scene_id, instructions)

        # FREE GPU MEMORY
        instr_gen.unload()
        print("  ✅ LLM unloaded, GPU memory freed")

        # ─────────────────────────────────────────────────────────
        # STEP 3: Evaluation (API call, no GPU needed)
        # ─────────────────────────────────────────────────────────
        print("\n[STEP 3] Evaluating with GPT-4o API...")
        from m03_evaluator import InstructionEvaluator

        evaluator = InstructionEvaluator(use_openai=True)

        all_metrics = []
        for scene_id in pairs.keys():
            eval_results = evaluator.evaluate_scene(scene_id, self.io)
            self.io.save_evaluation(scene_id, eval_results)

            # Collect for CSV
            all_metrics.append(self.io.flatten_metrics(scene_id, eval_results))
            print(f"  {scene_id}: avg score = {eval_results.get('level_1_avg', {}).get('object_accuracy', 'N/A')}")

        # Save aggregated metrics to CSV
        self.io.save_metrics_csv(all_metrics)
        print("\n✅ Pipeline complete. Results in outputs/")

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--input_dir", type=str, default="data/images")
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator()

    if args.sanity:
        # Run on 1 sample
        pass
    elif args.full:
        orchestrator.run_full_pipeline(args.input_dir)

---
📄 Utils: src/utils/io_handler.py

"""
JSON/CSV file handler for storing pipeline outputs.
Standard practice for small-scale experiments (~100 samples).
"""

import json
import csv
from pathlib import Path
from datetime import datetime

class IOHandler:
    """File-based storage using JSON for structured data, CSV for metrics."""

    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.scenes_dir = self.output_dir / "m01_scenes"
        self.instructions_dir = self.output_dir / "m02_instructions"
        self.evaluations_dir = self.output_dir / "m03_evaluations"
        # NOTE: Directories created lazily by _ensure_dir() when saving

    def _ensure_dir(self, dir_path: Path):
        """Create directory if it doesn't exist (called before each save)."""
        dir_path.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # SCENE OPERATIONS (m01)
    # ─────────────────────────────────────────────────────────────────
    def save_scene(self, scene_id: str, paths: dict, scene_data: dict):
        """Save scene analysis to JSON. Creates outputs/m01_scenes/ if needed."""
        self._ensure_dir(self.scenes_dir)
        data = {
            "scene_id": scene_id,
            "isometric_path": str(paths["isometric"]),
            "topdown_path": str(paths["topdown"]),
            "objects": scene_data["objects"],
            "scene_type": scene_data["scene_type"],
            "layout_description": scene_data["layout_description"],
            "timestamp": datetime.now().isoformat()
        }
        filepath = self.scenes_dir / f"{scene_id}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_scene(self, scene_id: str) -> dict:
        """Load scene data from JSON."""
        filepath = self.scenes_dir / f"{scene_id}.json"
        with open(filepath, "r") as f:
            return json.load(f)

    def list_scenes(self) -> list:
        """List all processed scene IDs."""
        return [f.stem for f in self.scenes_dir.glob("*.json")]

    # ─────────────────────────────────────────────────────────────────
    # INSTRUCTION OPERATIONS (m02)
    # ─────────────────────────────────────────────────────────────────
    def save_instructions(self, scene_id: str, instructions: dict):
        """Save generated instructions to JSON. Creates outputs/m02_instructions/ if needed."""
        self._ensure_dir(self.instructions_dir)
        data = {
            "scene_id": scene_id,
            "level_1": instructions.get("level_1", []),
            "level_2": instructions.get("level_2", []),
            "level_3": instructions.get("level_3", []),
            "timestamp": datetime.now().isoformat()
        }
        filepath = self.instructions_dir / f"{scene_id}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_instructions(self, scene_id: str) -> list:
        """Load instructions as flat list with level info."""
        filepath = self.instructions_dir / f"{scene_id}.json"
        with open(filepath, "r") as f:
            data = json.load(f)

        # Flatten to list format for evaluator
        result = []
        for level in [1, 2, 3]:
            for instr in data.get(f"level_{level}", []):
                result.append({"text": instr, "level": level})
        return result

    # ─────────────────────────────────────────────────────────────────
    # EVALUATION OPERATIONS (m03)
    # ─────────────────────────────────────────────────────────────────
    def save_evaluation(self, scene_id: str, eval_results: dict):
        """Save evaluation results to JSON. Creates outputs/m03_evaluations/ if needed."""
        self._ensure_dir(self.evaluations_dir)
        eval_results["timestamp"] = datetime.now().isoformat()
        filepath = self.evaluations_dir / f"{scene_id}.json"
        with open(filepath, "w") as f:
            json.dump(eval_results, f, indent=2)

    def load_evaluation(self, scene_id: str) -> dict:
        """Load evaluation results from JSON."""
        filepath = self.evaluations_dir / f"{scene_id}.json"
        with open(filepath, "r") as f:
            return json.load(f)

    # ─────────────────────────────────────────────────────────────────
    # METRICS CSV (for pandas analysis)
    # ─────────────────────────────────────────────────────────────────
    def flatten_metrics(self, scene_id: str, eval_results: dict) -> dict:
        """Flatten evaluation results to single row for CSV."""
        row = {"scene_id": scene_id}
        for level in [1, 2, 3]:
            avg = eval_results.get(f"level_{level}_avg", {})
            for metric in ["object_accuracy", "spatial_coherence", "task_clarity",
                          "difficulty_alignment", "executability"]:
                row[f"L{level}_{metric}"] = avg.get(metric, None)
        return row

    def save_metrics_csv(self, metrics_list: list):
        """Save all metrics to single CSV (pandas-friendly). Creates outputs/ if needed."""
        if not metrics_list:
            return

        self._ensure_dir(self.output_dir)
        filepath = self.output_dir / "metrics.csv"
        fieldnames = list(metrics_list[0].keys())

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_list)

    def load_metrics_csv(self):
        """Load metrics CSV as list of dicts (or use pandas directly)."""
        filepath = self.output_dir / "metrics.csv"
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)

---
📊 Output Files Summary

┌─────────────────────────────────┬───────────────────────────────────────────────────┐
│           File/Dir              │                    Purpose                        │
├─────────────────────────────────┼───────────────────────────────────────────────────┤
│ outputs/m01_scenes/*.json       │ Object detection: objects, scene_type, layout     │
├─────────────────────────────────┼───────────────────────────────────────────────────┤
│ outputs/m02_instructions/*.json │ Generated instructions per level                  │
├─────────────────────────────────┼───────────────────────────────────────────────────┤
│ outputs/m03_evaluations/*.json  │ VLM-as-Judge scores per instruction               │
├─────────────────────────────────┼───────────────────────────────────────────────────┤
│ outputs/metrics.csv             │ Aggregated scores for pandas analysis             │
└─────────────────────────────────┴───────────────────────────────────────────────────┘

Example JSON files:

# outputs/m01_scenes/scene_001.json
{
  "scene_id": "scene_001",
  "isometric_path": "data/images/scene_001_isometric.png",
  "topdown_path": "data/images/scene_001_topdown.png",
  "objects": [
    {"name": "bench", "confidence": 0.92},
    {"name": "fountain", "confidence": 0.88}
  ],
  "scene_type": "outdoor_mall",
  "layout_description": "Open plaza with central fountain"
}

# outputs/metrics.csv (pandas-friendly)
scene_id,L1_object_accuracy,L1_spatial_coherence,...,L3_executability
scene_001,4.5,4.2,...,4.0
scene_002,4.8,4.6,...,4.3

---
🔄 Execution Flow

# 1. Sanity check (1 sample)
python -u src/m04_pipeline_orchestrator.py --sanity 2>&1 | tee logs/m04_sanity.log

# 2. Full run
python -u src/m04_pipeline_orchestrator.py --full --input_dir Literature/Prev_work2/dataset/3DReasoningProject_images 2>&1 | tee logs/m04_full.log

# 3. Evaluation only (if instructions already generated)
python -u src/m04_pipeline_orchestrator.py --eval_only 2>&1 | tee logs/m04_eval.log

# Quick analysis in Python:
import pandas as pd
df = pd.read_csv("outputs/metrics.csv")
print(df.describe())
