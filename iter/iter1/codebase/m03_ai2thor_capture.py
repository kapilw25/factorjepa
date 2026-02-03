"""
AI2-THOR Image Capture (Top-Down Only)

Usage (on GPU Server):
    python -u src/m03_ai2thor_capture.py --scenes 50 | tee logs/log1.log         # 50 scenes (default)
    python -u src/m03_ai2thor_capture.py --scenes 120 | tee logs/log2.log        # All 120 scenes
    python -u src/m03_ai2thor_capture.py --metadata-only | tee logs/log3.log     # Metadata only (no images)
    python -u src/m03_ai2thor_capture.py --scene FloorPlan1 | tee logs/log4.log  # Single scene
    python -u src/m03_ai2thor_capture.py --list | tee logs/log5.log              # List available scenes

Usage (on M1 Mac - no GPU, use existing images):
    python -u src/m03_ai2thor_capture.py --grid-only | tee logs/log6.log         # Add grid to existing images
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

# AI2-THOR import (only available on GPU server)
try:
    from ai2thor.controller import Controller
    AI2THOR_AVAILABLE = True
except ImportError:
    AI2THOR_AVAILABLE = False
    print("Warning: ai2thor not installed. Install with: pip install ai2thor")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "images" / "ai2thor"


def draw_grid_overlay(img, grid_size: tuple):
    """
    Draw grid overlay on a PIL Image. Reusable by both capture and --grid-only.

    Args:
        img: PIL Image object
        grid_size: Tuple of (rows, cols)

    Returns:
        PIL Image with grid overlay
    """
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    rows, cols = grid_size
    img_width, img_height = img.size

    cell_width = img_width / cols
    cell_height = img_height / rows

    # Draw vertical lines (red)
    for i in range(cols + 1):
        x = int(i * cell_width)
        draw.line([(x, 0), (x, img_height)], fill=(255, 0, 0), width=1)

    # Draw horizontal lines (red)
    for i in range(rows + 1):
        y = int(i * cell_height)
        draw.line([(0, y), (img_width, y)], fill=(255, 0, 0), width=1)

    # Add grid size label
    draw.text((10, 10), f"Grid: {rows}x{cols}", fill=(255, 255, 0))

    return img


def get_available_scenes() -> Dict[str, List[str]]:
    """Get available AI2-THOR scene categories."""
    return {
        # iTHOR (120 scenes)
        "kitchens": [f"FloorPlan{i}" for i in range(1, 31)],
        "living_rooms": [f"FloorPlan{i}" for i in range(201, 231)],
        "bedrooms": [f"FloorPlan{i}" for i in range(301, 331)],
        "bathrooms": [f"FloorPlan{i}" for i in range(401, 431)],
        # RoboTHOR (89 apartments) - uses different naming
        # "robothor": [f"FloorPlan_Train{i}_{j}" for i in range(1, 13) for j in range(1, 6)],
    }


def get_all_ithor_scenes() -> List[str]:
    """Get all 120 iTHOR scenes."""
    scenes = get_available_scenes()
    all_scenes = []
    for category in ["kitchens", "living_rooms", "bedrooms", "bathrooms"]:
        all_scenes.extend(scenes[category])
    return all_scenes


def get_poc_scenes(n: int = 50) -> List[str]:
    """
    Get n sample scenes for POC (balanced mix of room types).

    Distribution for n scenes:
    - 25% Kitchens (FloorPlan1-30)
    - 25% Living Rooms (FloorPlan201-230)
    - 25% Bedrooms (FloorPlan301-330)
    - 25% Bathrooms (FloorPlan401-430)
    """
    if n >= 120:
        return get_all_ithor_scenes()

    # Balanced distribution across 4 room types
    per_type = max(1, n // 4)
    remainder = n % 4

    scenes = []

    # Kitchens: FloorPlan1-30
    kitchen_count = per_type + (1 if remainder > 0 else 0)
    scenes.extend([f"FloorPlan{i}" for i in range(1, min(31, kitchen_count + 1))])

    # Living Rooms: FloorPlan201-230
    living_count = per_type + (1 if remainder > 1 else 0)
    scenes.extend([f"FloorPlan{i}" for i in range(201, min(231, 201 + living_count))])

    # Bedrooms: FloorPlan301-330
    bedroom_count = per_type + (1 if remainder > 2 else 0)
    scenes.extend([f"FloorPlan{i}" for i in range(301, min(331, 301 + bedroom_count))])

    # Bathrooms: FloorPlan401-430
    bathroom_count = per_type
    scenes.extend([f"FloorPlan{i}" for i in range(401, min(431, 401 + bathroom_count))])

    return scenes[:n]


class AI2THORCapture:
    """Capture images and metadata from AI2-THOR scenes."""

    def __init__(self, width: int = 600, height: int = 600):
        if not AI2THOR_AVAILABLE:
            raise RuntimeError("ai2thor not installed")

        self.width = width
        self.height = height
        self.controller = None

    def start(self):
        """Initialize AI2-THOR controller."""
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene="FloorPlan1",
            gridSize=0.25,
            snapToGrid=True,
            rotateStepDegrees=90,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=self.width,
            height=self.height,
            fieldOfView=90,
        )
        print("AI2-THOR controller initialized")

    def stop(self):
        """Stop AI2-THOR controller."""
        if self.controller:
            self.controller.stop()
            print("AI2-THOR controller stopped")

    def capture_scene(self, scene_name: str, output_dir: Path, metadata_only: bool = False) -> Dict:
        """
        Capture images and metadata for a single scene.

        Args:
            scene_name: Scene name (e.g., FloorPlan1)
            output_dir: Output directory
            metadata_only: If True, skip image capture (faster)

        Returns:
            Dict with scene metadata
        """
        # Reset to scene
        self.controller.reset(scene=scene_name)

        # Get scene metadata
        event = self.controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]

        # Get objects in scene
        objects = self._get_scene_objects()

        # Calculate grid bounds
        grid_info = self._calculate_grid_info(reachable_positions)

        # Create output directory
        scene_dir = output_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = {
            "scene_name": scene_name,
            "grid_size": grid_info["grid_size"],
            "grid_bounds": grid_info["bounds"],
            "reachable_count": len(reachable_positions),
            "objects": objects,
            "images": {}
        }

        # Capture top-down image only (first-person removed - unsuitable for VLM navigation)
        if not metadata_only:
            top_down_img = self._capture_top_down()

            # Save original image
            top_down_path = scene_dir / "top_down.png"
            self._save_image(top_down_img, top_down_path)

            # Save grid-overlayed image (using dynamic grid size)
            top_down_grid_path = scene_dir / "top_down_grid.png"
            self._save_image_with_grid(top_down_img, top_down_grid_path, grid_info["grid_size"])

            metadata["images"] = {
                "top_down": str(top_down_path.relative_to(PROJECT_ROOT)),
                "top_down_grid": str(top_down_grid_path.relative_to(PROJECT_ROOT)),
            }

        # Save metadata
        metadata_path = scene_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        mode = "metadata" if metadata_only else "full"
        print(f"  [{mode}] {scene_name}: grid={grid_info['grid_size']}, objects={len(objects)}")

        return metadata

    def _get_scene_objects(self) -> List[Dict]:
        """Get list of objects in current scene."""
        objects = []
        for obj in self.controller.last_event.metadata["objects"]:
            if obj["pickupable"] or obj["receptacle"]:
                objects.append({
                    "name": obj["name"],
                    "type": obj["objectType"],
                    "position": {
                        "x": round(obj["position"]["x"], 2),
                        "y": round(obj["position"]["y"], 2),
                        "z": round(obj["position"]["z"], 2),
                    },
                    "pickupable": obj["pickupable"],
                    "receptacle": obj["receptacle"],
                })
        return objects

    def _calculate_grid_info(self, positions: List[Dict]) -> Dict:
        """Calculate grid info from reachable positions."""
        if not positions:
            return {"grid_size": (0, 0), "bounds": {}}

        x_coords = [p["x"] for p in positions]
        z_coords = [p["z"] for p in positions]

        min_x, max_x = min(x_coords), max(x_coords)
        min_z, max_z = min(z_coords), max(z_coords)

        # Estimate grid size (assuming 0.25 step size)
        step = 0.25
        rows = int((max_z - min_z) / step) + 1
        cols = int((max_x - min_x) / step) + 1

        return {
            "grid_size": (rows, cols),
            "bounds": {
                "min_x": round(min_x, 2),
                "max_x": round(max_x, 2),
                "min_z": round(min_z, 2),
                "max_z": round(max_z, 2),
            }
        }

    def _capture_fallback(self):
        """Fallback capture if ToggleMapView fails."""
        event = self.controller.step(action="Pass")
        return event.frame

    def _capture_top_down(self):
        """Capture top-down (bird's eye) view."""
        event = self.controller.step(action="ToggleMapView")
        if event.metadata["lastActionSuccess"]:
            return event.frame
        return self._capture_fallback()

    def _save_image(self, frame, path: Path):
        """Save numpy frame as PNG image."""
        try:
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(path)
        except ImportError:
            print("Warning: Pillow not installed. Cannot save images.")

    def _save_image_with_grid(self, frame, path: Path, grid_size: tuple):
        """Save image with grid overlay using dynamic grid size from metadata."""
        try:
            from PIL import Image
            img = Image.fromarray(frame)
            img = draw_grid_overlay(img, grid_size)
            img.save(path)
        except ImportError:
            print("Warning: Pillow not installed. Cannot save grid images.")


def add_grid_to_existing(output_dir: Path):
    """
    Add grid overlay to existing top_down.png images (no AI2-THOR required).
    Uses draw_grid_overlay() and reads grid_size from metadata.json.
    """
    from PIL import Image

    print(f"\nAdding grid overlays to existing images...")
    print(f"Directory: {output_dir}\n")

    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        return

    scene_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    processed, skipped = 0, 0

    for scene_dir in scene_dirs:
        top_down = scene_dir / "top_down.png"
        metadata_file = scene_dir / "metadata.json"
        grid_output = scene_dir / "top_down_grid.png"

        if not top_down.exists() or not metadata_file.exists():
            skipped += 1
            continue

        # Load grid size from metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        grid_size = tuple(metadata.get("grid_size", (10, 10)))

        # Apply grid overlay and save
        img = Image.open(top_down)
        img = draw_grid_overlay(img.copy(), grid_size)
        img.save(grid_output)

        processed += 1
        if processed % 20 == 0:
            print(f"  Progress: {processed} scenes")

    print(f"\nComplete: {processed} processed, {skipped} skipped")


def capture_scenes(scene_names: List[str], output_dir: Path, metadata_only: bool = False):
    """Capture multiple scenes."""
    mode = "metadata only" if metadata_only else "full (images + metadata)"
    print(f"\nCapturing {len(scene_names)} scenes [{mode}]...")
    print(f"Output directory: {output_dir}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = AI2THORCapture()
    capture.start()

    results = []
    for i, scene in enumerate(scene_names):
        try:
            metadata = capture.capture_scene(scene, output_dir, metadata_only)
            results.append(metadata)
        except Exception as e:
            print(f"  Error capturing {scene}: {e}")

        # Progress for large batches
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(scene_names)}")

    capture.stop()

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "scenes_captured": len(results),
            "metadata_only": metadata_only,
            "scenes": [r["scene_name"] for r in results],
        }, f, indent=2)

    print(f"\nCapture complete: {len(results)}/{len(scene_names)} scenes")
    print(f"Summary saved to: {summary_path}")

    return results


def list_scenes():
    """List all available AI2-THOR scenes."""
    scenes = get_available_scenes()
    print("\nAvailable AI2-THOR Scenes (iTHOR):")
    print("-" * 40)

    total = 0
    for category, scene_list in scenes.items():
        print(f"\n{category.upper()} ({len(scene_list)} scenes):")
        print(f"  {scene_list[0]} - {scene_list[-1]}")
        total += len(scene_list)

    print("\n" + "-" * 40)
    print(f"Total iTHOR scenes: {total}")

    # Show distribution for default 50 scenes
    poc_50 = get_poc_scenes(50)
    kitchens = [s for s in poc_50 if int(s.replace("FloorPlan", "")) < 100]
    living = [s for s in poc_50 if 200 <= int(s.replace("FloorPlan", "")) < 300]
    bedrooms = [s for s in poc_50 if 300 <= int(s.replace("FloorPlan", "")) < 400]
    bathrooms = [s for s in poc_50 if 400 <= int(s.replace("FloorPlan", "")) < 500]

    print(f"\nDefault --scenes 50 distribution:")
    print(f"  Kitchens:     {len(kitchens)} ({kitchens[0]}-{kitchens[-1]})")
    print(f"  Living Rooms: {len(living)} ({living[0]}-{living[-1]})")
    print(f"  Bedrooms:     {len(bedrooms)} ({bedrooms[0]}-{bedrooms[-1]})")
    print(f"  Bathrooms:    {len(bathrooms)} ({bathrooms[0]}-{bathrooms[-1]})")


def main():
    parser = argparse.ArgumentParser(
        description="Capture AI2-THOR images for VLM navigation"
    )
    parser.add_argument(
        "--scenes", type=int, default=50,
        help="Number of scenes to capture with images (default: 50, max: 120)"
    )
    parser.add_argument(
        "--scene", type=str,
        help="Capture a single specific scene (e.g., FloorPlan1)"
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Capture metadata for all 120 iTHOR scenes (no images)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available scenes"
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_DIR),
        help="Output directory for images"
    )
    parser.add_argument(
        "--grid-only", action="store_true",
        help="Add grid overlay to existing images (no AI2-THOR required, runs on M1 Mac)"
    )

    args = parser.parse_args()

    if args.list:
        list_scenes()
        return

    output_dir = Path(args.output)

    # --grid-only: No AI2-THOR needed, works on M1 Mac
    if args.grid_only:
        add_grid_to_existing(output_dir)
        return

    if not AI2THOR_AVAILABLE:
        print("\nError: ai2thor not installed.")
        print("Install on GPU server with: pip install ai2thor")
        print("\nUse --list to see available scenes without ai2thor.")
        print("Use --grid-only to add grid overlays to existing images.")
        return

    if args.metadata_only:
        # Capture metadata for all 120 iTHOR scenes
        scenes = get_all_ithor_scenes()
        capture_scenes(scenes, output_dir, metadata_only=True)
    elif args.scene:
        # Capture single scene with images
        capture_scenes([args.scene], output_dir, metadata_only=False)
    else:
        # Capture POC scenes with images
        scenes = get_poc_scenes(args.scenes)
        capture_scenes(scenes, output_dir, metadata_only=False)


if __name__ == "__main__":
    main()
