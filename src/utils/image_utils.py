"""
Image utilities for 3D Scene Understanding pipeline.
Handles image pair finding, resizing, and base64 encoding.

Usage:
    from utils.image_utils import find_image_pairs, resize_image, encode_image_base64
    pairs = find_image_pairs("data/images")
    img = resize_image("path/to/image.png", (1024, 1024))
"""

import base64
import re
from pathlib import Path
from typing import Dict, Tuple, Optional
from PIL import Image


def find_image_pairs(input_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Find all isometric/topdown image pairs in directory (recursive).

    Expected naming convention:
        {scene_id}_isometric.png
        {scene_id}_topdown.png

    Example:
        1_Office.3_isometric.png + 1_Office.3_topdown.png
        -> scene_id = "1_Office.3"

    Args:
        input_dir: Root directory containing images (searches recursively)

    Returns:
        Dict mapping scene_id to {"isometric": path, "topdown": path}
    """
    input_path = Path(input_dir)
    pairs = {}

    # Find all isometric images recursively
    for iso_file in input_path.rglob("*_isometric.png"):
        # Extract scene_id by removing _isometric.png suffix
        scene_id = iso_file.stem.replace("_isometric", "")

        # Look for matching topdown image in same directory
        topdown_file = iso_file.parent / f"{scene_id}_topdown.png"

        if topdown_file.exists():
            pairs[scene_id] = {
                "isometric": str(iso_file),
                "topdown": str(topdown_file)
            }

    return pairs


def resize_image(image_path: str, size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """
    Load and resize image to specified dimensions.

    Args:
        image_path: Path to image file
        size: Target (width, height), default (1024, 1024)

    Returns:
        PIL Image object resized to target dimensions
    """
    img = Image.open(image_path)

    # Convert to RGB if needed (handles RGBA, grayscale, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize with high-quality resampling
    img = img.resize(size, Image.Resampling.LANCZOS)

    return img


def encode_image_base64(image_path: str, resize_to: Optional[Tuple[int, int]] = None) -> str:
    """
    Encode image to base64 string for API calls (e.g., OpenAI GPT-4o).

    Args:
        image_path: Path to image file
        resize_to: Optional (width, height) to resize before encoding

    Returns:
        Base64-encoded string of the image
    """
    img = Image.open(image_path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    if resize_to:
        img = img.resize(resize_to, Image.Resampling.LANCZOS)

    # Save to bytes buffer
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def get_image_info(image_path: str) -> Dict:
    """
    Get basic image information.

    Args:
        image_path: Path to image file

    Returns:
        Dict with width, height, mode, format
    """
    img = Image.open(image_path)
    return {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "format": img.format
    }
