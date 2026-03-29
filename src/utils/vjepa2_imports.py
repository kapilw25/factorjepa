"""
Import vjepa2 modules from deps/vjepa2/ without src.utils namespace collision.

Problem: vjepa2 uses `from src.models...` and `from src.utils.tensors...`.
Our project also has `src/utils/` with __init__.py, which shadows vjepa2's
`src/utils/` because Python merges CWD/src/ into the namespace package.

Solution: Temporarily change CWD and isolate sys.path/modules so Python
only sees vjepa2's `src/` tree. After importing, vjepa2 modules stay
cached in sys.modules and our environment is restored.
"""
import importlib
import os
import sys
from pathlib import Path

VJEPA2_ROOT = Path(__file__).parent.parent.parent / "deps" / "vjepa2"

_loaded = False


def _ensure_loaded():
    """One-time import of vjepa2 modules into sys.modules."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    vjepa2_root = str(VJEPA2_ROOT)

    # Save state
    saved_cwd = os.getcwd()
    saved_path = sys.path[:]
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "src" or key.startswith("src."):
            saved_modules[key] = sys.modules.pop(key)

    # Isolate: CWD to /tmp (no src/ there), path to vjepa2 only
    os.chdir("/tmp")
    sys.path = [vjepa2_root] + [
        p for p in saved_path
        if "factorjepa/src" not in p
    ]
    importlib.invalidate_caches()

    try:
        importlib.import_module("src.utils.tensors")
        importlib.import_module("src.models.utils.modules")
        importlib.import_module("src.models.utils.patch_embed")
        importlib.import_module("src.models.utils.pos_embs")
        importlib.import_module("src.models.vision_transformer")
        importlib.import_module("src.models.predictor")
        importlib.import_module("src.masks.utils")
        importlib.import_module("src.masks.multiseq_multiblock3d")
    finally:
        os.chdir(saved_cwd)
        sys.path = saved_path
        # Restore our src and src.utils but keep all vjepa2 src.models.* etc
        for key in ["src", "src.utils"]:
            if key in saved_modules:
                sys.modules[key] = saved_modules[key]
        importlib.invalidate_caches()


def get_vit_giant_xformers():
    _ensure_loaded()
    return sys.modules["src.models.vision_transformer"].vit_giant_xformers


def get_vit_predictor():
    _ensure_loaded()
    return sys.modules["src.models.predictor"].vit_predictor


def get_mask_generator():
    _ensure_loaded()
    return sys.modules["src.masks.multiseq_multiblock3d"]._MaskGenerator


def get_apply_masks():
    _ensure_loaded()
    return sys.modules["src.masks.utils"].apply_masks
