"""Generic embedding/array checkpoint save/load — atomic, corruption-safe.

Replaces duplicated save_checkpoint/load_checkpoint across m04/m04d/m05/m06c.
Use for any script that accumulates (N, D) arrays + string keys and needs
to resume after interrupt.

USAGE:
    from utils.checkpoint import save_embedding_checkpoint, load_embedding_checkpoint

    ckpt = output_dir / "checkpoint.npz"
    embeddings, keys, n = load_embedding_checkpoint(ckpt)  # Returns [], [], 0 if missing
    # ... accumulate more
    save_embedding_checkpoint(embeddings, keys, ckpt)       # Atomic write

For JSON-based checkpoints (tags, metadata): use save_json_checkpoint / load_json_checkpoint.
"""
import json
import os
from pathlib import Path

import numpy as np


def save_embedding_checkpoint(embeddings_list: list, keys_list: list,
                              checkpoint_file: Path) -> None:
    """Atomic checkpoint save: embeddings + keys to .npz.

    Writes to {stem}_tmp.npz first, then os.replace() for crash safety.
    No-op if embeddings_list is empty.

    Args:
        embeddings_list: List of 1D numpy arrays (each shape (D,))
        keys_list: List of string keys aligned with embeddings_list
        checkpoint_file: Target .npz path
    """
    if not embeddings_list:
        return
    checkpoint_file = Path(checkpoint_file)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = checkpoint_file.with_name(checkpoint_file.stem + "_tmp.npz")
    np.savez(tmp_file,
             embeddings=np.stack(embeddings_list).astype(np.float32),
             keys=np.array(keys_list, dtype=object))
    os.replace(tmp_file, checkpoint_file)


def load_embedding_checkpoint(checkpoint_file: Path) -> tuple:
    """Load checkpoint. Returns (embeddings_list, keys_list, count).

    Returns ([], [], 0) if file missing or corrupt. Never raises.

    Args:
        checkpoint_file: .npz path

    Returns:
        (embeddings_list, keys_list, count): Lists of arrays/strings + count
    """
    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.exists():
        return [], [], 0
    try:
        data = np.load(checkpoint_file, allow_pickle=True)
        emb_list = list(data["embeddings"])
        keys_list = list(data["keys"])
        print(f"Checkpoint loaded: {len(emb_list):,} items from {checkpoint_file.name}")
        return emb_list, keys_list, len(emb_list)
    except Exception as e:
        print(f"  WARN: checkpoint corrupt ({e}), starting fresh")
        return [], [], 0


def save_array_checkpoint(array: np.ndarray, checkpoint_file: Path) -> None:
    """Atomic save of a single numpy array (no keys). Use for projected embeddings,
    intermediate features, etc. Writes .npy atomically via tmp + os.replace().

    Suffix order: tmp ends in `.npy` (NOT `.npy.tmp`) because np.save auto-appends
    `.npy` when the filename doesn't already end in it (per numpy docs). If we
    wrote to `....npy.tmp`, numpy would silently rename to `....npy.tmp.npy` and
    the subsequent os.replace(tmp, final) would fail. errors_N_fixes.md #82.
    """
    checkpoint_file = Path(checkpoint_file)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = checkpoint_file.with_suffix(".tmp" + checkpoint_file.suffix)
    np.save(tmp_file, array)
    os.replace(tmp_file, checkpoint_file)


def save_json_checkpoint(data: dict | list, checkpoint_file: Path) -> None:
    """Atomic JSON checkpoint save. Use for tags, metadata, config summaries.
    Suffix order matches save_array_checkpoint for consistency (json.dump uses a
    file handle so isn't affected by numpy's auto-append, but the symmetric
    pattern keeps the codebase free of the #82 anti-pattern).
    """
    checkpoint_file = Path(checkpoint_file)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = checkpoint_file.with_suffix(".tmp" + checkpoint_file.suffix)
    with open(tmp_file, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, checkpoint_file)


def load_json_checkpoint(checkpoint_file: Path, default=None):
    """Load JSON checkpoint. Returns `default` if missing or corrupt."""
    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.exists():
        return default if default is not None else {}
    try:
        with open(checkpoint_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARN: JSON checkpoint corrupt ({e}), using default")
        return default if default is not None else {}
