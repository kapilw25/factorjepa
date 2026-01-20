"""
JSON/CSV file handler for storing pipeline outputs.
Standard practice for small-scale experiments (~100 samples).

Usage:
    from utils.io_handler import IOHandler
    io = IOHandler("outputs")
    io.save_scene(scene_id, paths, scene_data)
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class IOHandler:
    """File-based storage using JSON for structured data, CSV for metrics."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.scenes_dir = self.output_dir / "m01_scenes"
        self.instructions_dir = self.output_dir / "m02_instructions"
        self.evaluations_dir = self.output_dir / "m03_evaluations"
        # NOTE: Directories created lazily by _ensure_dir() when saving

    def _ensure_dir(self, dir_path: Path) -> None:
        """Create directory if it doesn't exist (called before each save)."""
        dir_path.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # SCENE OPERATIONS (m01)
    # ─────────────────────────────────────────────────────────────────
    def save_scene(self, scene_id: str, paths: Dict, scene_data: Dict) -> None:
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

    def load_scene(self, scene_id: str) -> Dict:
        """Load scene data from JSON."""
        filepath = self.scenes_dir / f"{scene_id}.json"
        with open(filepath, "r") as f:
            return json.load(f)

    def list_scenes(self) -> List[str]:
        """List all processed scene IDs."""
        if not self.scenes_dir.exists():
            return []
        return [f.stem for f in self.scenes_dir.glob("*.json")]

    # ─────────────────────────────────────────────────────────────────
    # INSTRUCTION OPERATIONS (m02)
    # ─────────────────────────────────────────────────────────────────
    def save_instructions(self, scene_id: str, instructions: Dict) -> None:
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

    def load_instructions(self, scene_id: str) -> List[Dict]:
        """Load instructions as flat list with level info."""
        filepath = self.instructions_dir / f"{scene_id}.json"
        with open(filepath, "r") as f:
            data = json.load(f)

        # Flatten to list format for evaluator
        result = []
        for level in [1, 2, 3]:
            for instr in data.get(f"level_{level}", []):
                if isinstance(instr, dict):
                    instr["level"] = level
                    result.append(instr)
                else:
                    result.append({"text": instr, "level": level})
        return result

    def list_instructions(self) -> List[str]:
        """List all scene IDs with generated instructions."""
        if not self.instructions_dir.exists():
            return []
        return [f.stem for f in self.instructions_dir.glob("*.json")]

    # ─────────────────────────────────────────────────────────────────
    # EVALUATION OPERATIONS (m03)
    # ─────────────────────────────────────────────────────────────────
    def save_evaluation(self, scene_id: str, eval_results: Dict) -> None:
        """Save evaluation results to JSON. Creates outputs/m03_evaluations/ if needed."""
        self._ensure_dir(self.evaluations_dir)
        eval_results["timestamp"] = datetime.now().isoformat()
        filepath = self.evaluations_dir / f"{scene_id}.json"
        with open(filepath, "w") as f:
            json.dump(eval_results, f, indent=2)

    def load_evaluation(self, scene_id: str) -> Dict:
        """Load evaluation results from JSON."""
        filepath = self.evaluations_dir / f"{scene_id}.json"
        with open(filepath, "r") as f:
            return json.load(f)

    def list_evaluations(self) -> List[str]:
        """List all scene IDs with evaluations."""
        if not self.evaluations_dir.exists():
            return []
        return [f.stem for f in self.evaluations_dir.glob("*.json")]

    # ─────────────────────────────────────────────────────────────────
    # METRICS CSV (for pandas analysis)
    # ─────────────────────────────────────────────────────────────────
    def flatten_metrics(self, scene_id: str, eval_results: Dict) -> Dict:
        """Flatten evaluation results to single row for CSV."""
        row = {"scene_id": scene_id}
        for level in [1, 2, 3]:
            avg = eval_results.get(f"level_{level}_avg", {})
            for metric in ["object_accuracy", "spatial_coherence", "task_clarity",
                          "difficulty_alignment", "executability"]:
                row[f"L{level}_{metric}"] = avg.get(metric, None)
        return row

    def save_metrics_csv(self, metrics_list: List[Dict]) -> None:
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

    def load_metrics_csv(self) -> List[Dict]:
        """Load metrics CSV as list of dicts (or use pandas directly)."""
        filepath = self.output_dir / "metrics.csv"
        if not filepath.exists():
            return []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)
