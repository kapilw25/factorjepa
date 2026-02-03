"""
VLM Agent for Navigation (Top-Down Only)

Usage:
    python -u src/m04_vlm_agent.py --image data/images/ai2thor/FloorPlan1/ | tee logs/log1.log
    python -u src/m04_vlm_agent.py --image data/images/ai2thor/FloorPlan1/ --task "refrigerator to stove" | tee logs/log2.log
    python -u src/m04_vlm_agent.py --image data/images/ai2thor/FloorPlan301/ --task "bed to desk" | tee logs/log3.log
"""

import os
import json
import base64
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPEN_API_KEY")


@dataclass
class VLMResponse:
    """Response from VLM agent."""
    path: List[str]
    raw_response: str
    success: bool
    error: Optional[str] = None


@dataclass
class ObjectIdentificationResponse:
    """Response from VLM object identification."""
    objects: List[Dict]
    raw_response: str
    success: bool
    error: Optional[str] = None


class VLMAgent:
    """
    Vision-Language Model agent for navigation.
    Uses GPT-4V to generate navigation paths from images.
    """

    VALID_DIRECTIONS = {"North", "South", "East", "West"}

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.api_key = OPENAI_API_KEY

        if not self.api_key:
            raise ValueError("OPEN_API_KEY not found in .env file")

    def generate_path(
        self,
        image_path: Path,
        start_object: str,
        goal_object: str
    ) -> VLMResponse:
        """
        Generate navigation path from top-down image.

        Args:
            image_path: Path to room image (PNG)
            start_object: Starting object name (e.g., "bed")
            goal_object: Goal object name (e.g., "lamp")

        Returns:
            VLMResponse with path and metadata
        """
        # Encode image
        base64_image = self._encode_image(image_path)

        # Build prompt (top-down only)
        prompt = self._build_prompt(start_object, goal_object)

        # Call GPT-4V
        try:
            response = self._call_gpt4v(base64_image, prompt)
            path = self._parse_path(response)

            return VLMResponse(
                path=path,
                raw_response=response,
                success=True
            )

        except Exception as e:
            return VLMResponse(
                path=[],
                raw_response="",
                success=False,
                error=str(e)
            )

    def identify_objects(self, image_path: Path) -> ObjectIdentificationResponse:
        """
        Identify all objects visible in the top-down image.

        Args:
            image_path: Path to room image (PNG)

        Returns:
            ObjectIdentificationResponse with list of objects
        """
        base64_image = self._encode_image(image_path)

        prompt = """Analyze this top-down (bird's eye) view of a room.

TASK: List ALL objects/furniture you can identify in this image.

OUTPUT FORMAT (JSON only):
{
    "room_type": "kitchen/bedroom/bathroom/living_room",
    "objects": [
        {"name": "object_name", "position": "brief location description"},
        {"name": "object_name", "position": "brief location description"}
    ]
}

List every distinct object you can see:"""

        try:
            response = self._call_gpt4v(base64_image, prompt)
            objects = self._parse_objects(response)

            return ObjectIdentificationResponse(
                objects=objects,
                raw_response=response,
                success=True
            )

        except Exception as e:
            return ObjectIdentificationResponse(
                objects=[],
                raw_response="",
                success=False,
                error=str(e)
            )

    def _parse_objects(self, response: str) -> List[Dict]:
        """Parse object list from VLM response."""
        response = response.strip()

        # Handle markdown code blocks
        if "```" in response:
            parts = response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    response = part
                    break

        try:
            data = json.loads(response)
            if isinstance(data, dict):
                return data.get("objects", [])
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        return []

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _build_prompt(self, start_object: str, goal_object: str) -> str:
        """Build VLM prompt for top-down navigation."""
        return f"""Analyze this top-down (bird's eye) view of a room.

TASK: Generate step-by-step navigation directions from the {start_object} to the {goal_object}.

COORDINATE SYSTEM:
- North = moving UP in the image
- South = moving DOWN in the image
- East = moving RIGHT in the image
- West = moving LEFT in the image

First, locate both objects in the image. Then determine the path between them.

OUTPUT FORMAT (JSON only):
{{
    "start_location": "describe where {start_object} is",
    "goal_location": "describe where {goal_object} is",
    "path": ["direction1", "direction2", "direction3"],
    "reasoning": "brief explanation"
}}

Valid directions in path array: "North", "South", "East", "West"

Generate the navigation:"""

    def _call_gpt4v(self, base64_image: str, prompt: str) -> str:
        """Call GPT-4V API with image."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _parse_path(self, response: str) -> List[str]:
        """Parse path from VLM response."""
        response = response.strip()

        # Handle markdown code blocks
        if "```" in response:
            parts = response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{") or part.startswith("["):
                    response = part
                    break

        # Try to parse as JSON
        try:
            data = json.loads(response)

            # Handle object with "path" key
            if isinstance(data, dict):
                path = data.get("path", [])
            elif isinstance(data, list):
                path = data
            else:
                path = []

            # Validate directions
            valid_path = []
            for direction in path:
                if direction in self.VALID_DIRECTIONS:
                    valid_path.append(direction)
                else:
                    print(f"Warning: Invalid direction '{direction}' ignored")
            return valid_path

        except json.JSONDecodeError:
            pass

        # Fallback: try to extract directions from text
        path = []
        for direction in self.VALID_DIRECTIONS:
            idx = 0
            while True:
                found = response.find(direction, idx)
                if found == -1:
                    break
                path.append((found, direction))
                idx = found + len(direction)

        path.sort(key=lambda x: x[0])
        return [d for _, d in path]


def load_scene_metadata(scene_dir: Path) -> Optional[Dict]:
    """Load metadata.json from scene directory."""
    metadata_path = scene_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def run_demo():
    """Run demo with a sample or placeholder image."""
    print("=" * 60)
    print("VLM AGENT DEMO")
    print("=" * 60)

    # Check for sample images
    sample_dir = PROJECT_ROOT / "data" / "images" / "ai2thor"

    if sample_dir.exists():
        scenes = list(sample_dir.iterdir())
        scenes = [s for s in scenes if s.is_dir() and (s / "top_down.png").exists()]

        if scenes:
            scene = scenes[0]
            print(f"\nUsing scene: {scene.name}")

            agent = VLMAgent()

            # Load metadata to get objects
            metadata = load_scene_metadata(scene)
            if metadata and metadata.get("objects"):
                objects = metadata["objects"][:2]
                start = objects[0]["type"] if objects else "start"
                goal = objects[1]["type"] if len(objects) > 1 else "goal"
            else:
                start, goal = "start", "goal"

            # Test with top-down view
            top_down = scene / "top_down.png"
            print(f"\nTask: Navigate from {start} to {goal}")
            print(f"Image: {top_down}")

            result = agent.generate_path(top_down, start, goal)

            print(f"\nResult:")
            print(f"  Success: {result.success}")
            print(f"  Path: {result.path}")
            print(f"  Path Length: {len(result.path)}")

            if result.error:
                print(f"  Error: {result.error}")

            return result

    print("\nNo AI2-THOR images found.")
    print("Run on GPU server first: python src/m03_ai2thor_capture.py --scenes 5")
    print("\nOr provide an image with: python src/m03_vlm_agent.py --image <path>")


def main():
    parser = argparse.ArgumentParser(
        description="VLM Agent for navigation path generation"
    )
    parser.add_argument(
        "--image", type=str,
        help="Path to room image or scene directory"
    )
    parser.add_argument(
        "--start", type=str, default="start",
        help="Starting object (default: start)"
    )
    parser.add_argument(
        "--goal", type=str, default="goal",
        help="Goal object (default: goal)"
    )
    parser.add_argument(
        "--task", type=str,
        help="Task description (e.g., 'bed to lamp')"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="OpenAI model (default: gpt-4o)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run demo with sample image"
    )
    parser.add_argument(
        "--identify", action="store_true",
        help="Identify objects in image (run before navigation)"
    )

    args = parser.parse_args()

    if args.test:
        run_demo()
        return

    if not args.image:
        parser.print_help()
        print("\n--- Running Demo ---")
        run_demo()
        return

    # Handle image path
    image_path = Path(args.image)
    if image_path.is_dir():
        image_path = image_path / "top_down.png"

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    # Object identification mode
    if args.identify:
        print(f"\nVLM Object Identification")
        print(f"  Image: {image_path}")
        print(f"  Model: {args.model}")

        agent = VLMAgent(model=args.model)
        result = agent.identify_objects(image_path)

        print(f"\nResult:")
        print(f"  Success: {result.success}")

        if result.success:
            print(f"\nVLM Raw Response:\n{result.raw_response}")
            print(f"\nParsed Objects ({len(result.objects)}):")
            for obj in result.objects:
                name = obj.get("name", "unknown")
                pos = obj.get("position", "")
                print(f"  - {name}: {pos}")
        else:
            print(f"  Error: {result.error}")

        return

    # Parse task if provided
    if args.task:
        parts = args.task.lower().split(" to ")
        if len(parts) == 2:
            args.start = parts[0].strip()
            args.goal = parts[1].strip()

    agent = VLMAgent(model=args.model)

    # Step 1: Always identify objects first
    print(f"\n{'='*60}")
    print(f"STEP 1: Object Identification")
    print(f"{'='*60}")
    print(f"  Image: {image_path}")
    print(f"  Model: {args.model}")

    id_result = agent.identify_objects(image_path)

    if id_result.success:
        print(f"\nVLM Response:\n{id_result.raw_response}")
        print(f"\nParsed Objects ({len(id_result.objects)}):")
        for obj in id_result.objects:
            name = obj.get("name", "unknown")
            pos = obj.get("position", "")
            print(f"  - {name}: {pos}")
    else:
        print(f"  Error: {id_result.error}")

    # Step 2: Navigation (if task provided)
    if not args.task:
        print(f"\n{'='*60}")
        print(f"No --task provided. Use --task 'object1 to object2' for navigation.")
        return

    print(f"\n{'='*60}")
    print(f"STEP 2: Navigation")
    print(f"{'='*60}")
    print(f"  Task: {args.start} -> {args.goal}")

    result = agent.generate_path(image_path, args.start, args.goal)

    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Path: {result.path}")
    print(f"  Path Length: {len(result.path)}")

    # Always print raw response for debugging
    print(f"\nRaw VLM Response:\n{result.raw_response}")

    if result.error:
        print(f"  Error: {result.error}")


if __name__ == "__main__":
    main()
