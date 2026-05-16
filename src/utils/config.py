"""
Common configuration for WalkIndia-200k pipeline.
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
UTILS_DIR = SRC_DIR / "utils"
DATA_DIR = SRC_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
CLIPS_DIR = DATA_DIR / "clips"
SHARDS_DIR = DATA_DIR / "shards"
LOGS_DIR = PROJECT_ROOT / "logs"

# Consolidated output directories (all under outputs/)
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
OUTPUTS_DIR = OUTPUTS_ROOT / "full"
OUTPUTS_DATA_PREP_DIR = OUTPUTS_ROOT / "data_prep"
OUTPUTS_PROFILE_DIR = OUTPUTS_ROOT / "profile"

# Input data (moved from src/utils/ to configs/)
CONFIGS_DIR = PROJECT_ROOT / "configs"
YT_VIDEOS_JSON = CONFIGS_DIR / "YT_videos_raw.json"
TAG_TAXONOMY_JSON = CONFIGS_DIR / "tag_taxonomy.json"

# Ensure directories exist
for d in [VIDEOS_DIR, CLIPS_DIR, SHARDS_DIR, OUTPUTS_DIR, OUTPUTS_DATA_PREP_DIR,
          OUTPUTS_PROFILE_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# HuggingFace dataset config
HF_DATASET_REPO = "anonymousML123/walkindia-200k"

# HuggingFace private split repos (50GB limit per repo)
HF_DATASET_REPO_PART1 = "anonymousML123/walkindia-200k-part1"
HF_DATASET_REPO_PART2 = "anonymousML123/walkindia-200k-part2"
HF_DATASET_REPOS = [HF_DATASET_REPO_PART1, HF_DATASET_REPO_PART2]
HF_REPO_SIZE_LIMIT_GB = 300

# Re-encoding config (CRF 28 on 480p shrinks ~40-50%)
REENCODE_CRF = 28


def ensure_clips_exist() -> bool:
    """
    Check if clips exist locally. If not, download from HuggingFace (both split repos).
    Returns True if clips are available, False otherwise.
    """
    # Check if clips already exist (recursive search for hierarchical structure)
    clip_count = len(list(CLIPS_DIR.rglob("*.mp4"))) if CLIPS_DIR.exists() else 0

    if clip_count > 0:
        print(f"Found {clip_count} clips locally in {CLIPS_DIR}")
        return True

    # Auto-download from HuggingFace (try split repos, then fallback to single repo)
    repos_to_try = HF_DATASET_REPOS + [HF_DATASET_REPO]

    try:
        from huggingface_hub import snapshot_download

        for repo in repos_to_try:
            print(f"Downloading from HuggingFace: {repo}")
            try:
                local_dir = snapshot_download(
                    repo_id=repo,
                    repo_type="dataset",
                    local_dir=DATA_DIR / "hf_download",
                    allow_patterns=["clips/**/*.mp4"],
                )

                # Mirror entire clips tree (hierarchical structure)
                hf_clips_dir = Path(local_dir) / "clips"
                if hf_clips_dir.exists():
                    shutil.copytree(hf_clips_dir, CLIPS_DIR, dirs_exist_ok=True)

                # Cleanup temp download
                shutil.rmtree(DATA_DIR / "hf_download", ignore_errors=True)
                print(f"  Downloaded clips from {repo}")

            except Exception as e:
                print(f"  Skipping {repo}: {e}")
                continue

        # Verify
        clip_count = len(list(CLIPS_DIR.rglob("*.mp4")))
        print(f"Total clips available: {clip_count}")
        return clip_count > 0

    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

# All model/encoder configs loaded from YAML after function definitions below.
# See "Module-level constants from YAML" section.

# ── Pipeline config (clip limits from YAML, not hardcoded) ───────────
PIPELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"
_pipeline_cfg = None

def get_pipeline_config() -> dict:
    """Load configs/pipeline.yaml (cached). Single source of truth for clip limits."""
    global _pipeline_cfg
    if _pipeline_cfg is None:
        import yaml
        with open(PIPELINE_CONFIG_PATH) as f:
            _pipeline_cfg = yaml.safe_load(f)
    return _pipeline_cfg


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base. Overlay values win on conflict."""
    merged = base.copy()
    for k, v in overlay.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_train_config_with_extends(train_config: str) -> dict:
    """Load a single train yaml + resolve its `extends:` chain (no model/pipeline merge).

    iter13 v13 (2026-05-07): factored out of load_merged_config so callers like
    m10_sam_segment + m11_factor_datasets — which need ONLY the resolved train
    config (factor_datasets / interaction_mining blocks) and don't have a
    --model-config arg — can still see the full inheritance chain.

    Walks `extends: surgery_base.yaml → base_optimization.yaml` so a variant
    yaml's child fields override base, but base fields the variant doesn't
    mention stay inherited.

    Args:
        train_config: Path to train YAML (e.g., 'configs/train/surgery_3stage_DI_encoder.yaml').

    Returns:
        Resolved dict with full inheritance chain merged (variant wins on conflict).
    """
    train_path = Path(train_config)
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path

    if not train_path.exists():
        print(f"FATAL: Train config not found: {train_path}")
        sys.exit(1)

    with open(train_path) as f:
        train_cfg = yaml.safe_load(f)

    # Extends-chain resolution mirrors load_merged_config's loop. See full
    # comment there for the design rationale.
    seen = {train_path.resolve()}
    while True:
        extends = train_cfg.pop("extends", None)
        if not extends:
            break
        base_path = (train_path.parent / extends).resolve()
        if base_path in seen:
            print(f"FATAL: extends cycle detected at {base_path}")
            sys.exit(1)
        seen.add(base_path)
        if not base_path.exists():
            print(f"FATAL: extends target not found: {base_path}")
            sys.exit(1)
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        train_cfg = _deep_merge(base_cfg, train_cfg)
        train_path = base_path

    return train_cfg


def load_merged_config(model_config: str, train_config: str) -> dict:
    """Load and merge: pipeline.yaml (base) + model/*.yaml + train/*.yaml.

    Train configs that specify 'extends: base_optimization.yaml' are merged
    with that base first, then model config is overlaid.

    Args:
        model_config: Path to model YAML (e.g., 'configs/model/vjepa2_1.yaml')
        train_config: Path to train YAML (e.g., 'configs/legacy2/explora.yaml')

    Returns:
        Single merged dict with all config values.
    """
    model_path = Path(model_config)
    train_path = Path(train_config)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path

    if not model_path.exists():
        print(f"FATAL: Model config not found: {model_path}")
        sys.exit(1)
    if not train_path.exists():
        print(f"FATAL: Train config not found: {train_path}")
        sys.exit(1)

    pipeline_cfg = get_pipeline_config()
    with open(model_path) as f:
        model_cfg = yaml.safe_load(f)
    with open(train_path) as f:
        train_cfg = yaml.safe_load(f)

    # Handle 'extends' in train config (recursive — supports chains like
    # surgery_2stage_noDI_encoder.yaml → surgery_base.yaml → base_optimization.yaml).
    # iter11 v3 (2026-04-26): walk the chain until the parent has no `extends`
    # key. Each level overlays its child via _deep_merge (child wins on conflict).
    seen = {train_path.resolve()}
    while True:
        extends = train_cfg.pop("extends", None)
        if not extends:
            break
        base_path = (train_path.parent / extends).resolve()
        if base_path in seen:
            print(f"FATAL: extends cycle detected at {base_path}")
            sys.exit(1)
        seen.add(base_path)
        if not base_path.exists():
            print(f"FATAL: extends target not found: {base_path}")
            sys.exit(1)
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        train_cfg = _deep_merge(base_cfg, train_cfg)
        train_path = base_path  # next iteration resolves further `extends:` relative to the parent's dir

    # Merge order: pipeline (base) → model → train (train wins on conflict)
    merged = _deep_merge(pipeline_cfg, model_cfg)
    merged = _deep_merge(merged, train_cfg)

    # iter11 #51-followup: mirror model.{crop_size,patch_size,tubelet_size} into
    # data.* because legacy m09a/b/c + utils.training.producer_thread still read
    # cfg["data"][<key>] in several augment/producer code paths (errors_N_fixes #51
    # partial-renamed these but missed 6 occurrences). DRY reconciliation at merge
    # time — keeps a single canonical source (model config) while supporting both
    # access patterns without hunting every call-site.
    if "model" in merged:
        _model_section = merged["model"]
        _data_section = merged.setdefault("data", {})
        for _k in ("crop_size", "patch_size", "tubelet_size"):
            if _k in _model_section and _k not in _data_section:
                _data_section[_k] = _model_section[_k]

    return merged


def get_model_config(model_config: str = None) -> dict:
    """Load a model config YAML (standalone, no merge). Useful for m05 frozen eval."""
    if model_config is None:
        model_config = str(CONFIGS_DIR / "model" / "vjepa2_1.yaml")
    path = Path(model_config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"FATAL: Model config not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def add_model_config_arg(parser):
    """Add --model-config flag. Required (no default — FAIL LOUD per src/CLAUDE.md)."""
    parser.add_argument(
        "--model-config", required=True,
        help="Model config YAML (e.g., configs/model/vjepa2_1.yaml)")


def add_train_config_arg(parser):
    """Add --train-config flag. Required (no default — FAIL LOUD per src/CLAUDE.md)."""
    parser.add_argument(
        "--train-config", required=True,
        help="Training config YAML (e.g., configs/legacy2/explora.yaml)")


def get_sanity_clip_limit(module: str) -> int:
    """Get SANITY clip limit for a module from configs/pipeline.yaml."""
    cfg = get_pipeline_config()
    return cfg["sanity"].get(module, cfg["sanity"]["default"])


def get_poc_clip_limit(module: str) -> int:
    """Get POC clip limit for a module from configs/pipeline.yaml.

    iter13 v13 FIX-19 (2026-05-07): mirrors get_sanity_clip_limit pattern. POC
    sits between SANITY (n=20 — code correctness) and FULL (n=10K+ — paper),
    giving statistically meaningful per-clip quality distributions for ~10×
    less wall time than FULL. Per-module overrides in pipeline.yaml `poc:` block.
    """
    cfg = get_pipeline_config()
    return cfg["poc"].get(module, cfg["poc"]["default"])


def get_total_clips(local_data: str = None, subset_file: str = None) -> int:
    """Discover total clip count from data source. Never hardcode."""
    if subset_file:
        keys = load_subset(subset_file)
        return len(keys)
    if local_data:
        manifest = Path(local_data) / "manifest.json"
        if manifest.exists():
            data = json.load(open(manifest))
            # iter15 audit (2026-05-15): FAIL LOUD on malformed manifest (CLAUDE.md
            # "No .get(key, default)"). Manifest MUST have one of {n, n_clips,
            # total_clips}; if all 3 missing, raise instead of silently returning 0.
            for key in ("n", "n_clips", "total_clips"):
                if key in data:
                    return data[key]
            raise RuntimeError(
                f"FATAL get_total_clips: manifest {manifest} missing all of "
                f"{{n, n_clips, total_clips}}. Keys present: {list(data.keys())}")
    # iter15 audit (2026-05-15): caller passed neither subset_file nor local_data.
    # Previously returned 0 (silent) — now FAIL LOUD per CLAUDE.md.
    raise RuntimeError(
        "FATAL get_total_clips: called with neither subset_file nor local_data. "
        "Pass exactly one. Silent zero return removed (iter15 audit).")


# ═══════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS FROM YAML — all loaded AFTER function defs above.
# No hardcoded model IDs, dims, or magic numbers. Rule 15 in CLAUDE.md.
# ═══════════════════════════════════════════════════════════════════════

_pcfg = get_pipeline_config()

# V-JEPA config — from configs/model/ YAML (default: 2.1 ViT-G 2B)
_default_model_cfg = get_model_config()["model"]
VJEPA_MODEL_ID = _default_model_cfg["hf_model_id"]
VJEPA_EMBEDDING_DIM = _default_model_cfg["embed_dim"]
VJEPA_CHECKPOINT_PATH = _default_model_cfg["checkpoint_path"]
VJEPA_FRAMES_PER_CLIP = _pcfg["gpu"]["eval_frames_per_clip"]

# FAISS config
FAISS_K_NEIGHBORS = _pcfg["eval"]["faiss_k_neighbors"]

# Scene detection config
CLIP_MIN_DURATION = _pcfg["scene_detection"]["clip_min_duration"]
CLIP_MAX_DURATION = _pcfg["scene_detection"]["clip_max_duration"]

# VLM config
VLM_MODELS = _pcfg["vlm"]
QWEN_MODEL_ID = _pcfg["vlm"]["qwen"]

# Bake-off config
BAKEOFF_CLIP_COUNT = _pcfg["bakeoff"]["clips"]

# POC / output dirs
SUBSET_FILE = PROJECT_ROOT / "data" / "subset_10k.json"
OUTPUTS_SANITY_DIR = OUTPUTS_ROOT / "sanity"
OUTPUTS_POC_DIR = OUTPUTS_ROOT / "poc"
BAKEOFF_DIR = DATA_DIR / "bakeoff"

# Output files (legacy shortcuts — prefer get_encoder_files() for encoder-specific paths)
EMBEDDINGS_FILE = OUTPUTS_DIR / "embeddings.npy"
TAGS_FILE = OUTPUTS_DIR / "tags.json"
UMAP_PLOT_PNG = OUTPUTS_DIR / "m08_umap.png"
UMAP_PLOT_PDF = OUTPUTS_DIR / "m08_umap.pdf"
METRICS_FILE = OUTPUTS_DIR / "m06_metrics.json"


# ── Encoder registry — loaded from configs/pipeline.yaml → encoders section ──
# V-JEPA-derived encoders inherit dim from the default model config.
def _build_encoder_registry() -> dict:
    enc_cfg = _pcfg["encoders"]
    registry = {}
    for name, entry in enc_cfg.items():
        dim = entry["dim"] if entry["dim"] is not None else VJEPA_EMBEDDING_DIM
        model_id = entry["model_id"]
        registry[name] = {
            "model_id": model_id,
            "dim": dim,
            "type": entry["type"],
            "suffix": entry["suffix"],
        }
    return registry

ENCODER_REGISTRY = _build_encoder_registry()


def get_encoder_info(encoder: str) -> dict:
    """Get encoder info from registry, with dynamic fallback for unregistered encoders."""
    if encoder in ENCODER_REGISTRY:
        return ENCODER_REGISTRY[encoder]
    return {"model_id": None, "dim": VJEPA_EMBEDDING_DIM, "type": "video_adapted", "suffix": f"_{encoder}"}


def get_encoder_files(encoder: str, output_dir: Path) -> dict:
    """Return {embeddings, paths, metrics, knn_indices, umap_2d} paths for an encoder."""
    sfx = get_encoder_info(encoder)["suffix"]
    return {
        "embeddings":  output_dir / f"embeddings{sfx}.npy",
        "paths":       output_dir / f"embeddings{sfx}.paths.npy",
        "metrics":     output_dir / f"m06_metrics{sfx}.json",
        "knn_indices": output_dir / f"knn_indices{sfx}.npy",
        "umap_2d":     output_dir / f"umap_2d{sfx}.npy",
    }


def add_encoder_arg(parser):
    """Add --encoder argument to m05b/m06/m07/m08 parsers.

    Accepts registered encoders + any custom name (for Ch10 adapted variants).
    """
    parser.add_argument("--encoder", default="vjepa",
                        help=f"Encoder name (registered: {', '.join(ENCODER_REGISTRY.keys())}; "
                             "or any custom name for adapted variants)")


# Ensure POC/sanity/bakeoff directories exist
for d in [OUTPUTS_SANITY_DIR, OUTPUTS_POC_DIR, BAKEOFF_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SUBSET / POC UTILITIES
# =============================================================================

def load_subset(subset_path: str = None) -> set:
    """
    Load subset clip keys from JSON file.

    Args:
        subset_path: Path to subset JSON (e.g., data/subset_10k_local/subset_10k.json).
                     If None, returns empty set (= no filtering, full mode).

    Returns:
        Set of clip keys (e.g., {"goa/walking/04YKvC8kAgI/04YKvC8kAgI-000.mp4", ...})
        Empty set means no filtering (full mode).
    """
    if subset_path is None:
        return set()

    p = Path(subset_path)
    if not p.exists():
        print(f"ERROR: Subset file not found: {p}")
        sys.exit(1)

    with open(p) as f:
        data = json.load(f)

    keys = set(data["clip_keys"])
    print(f"[POC] Loaded subset: {len(keys):,} clip keys from {p.name}")
    return keys


def verify_npy_matches_subset(arr_or_path, subset_path: str, label: str = "embeddings"):
    """Fail-loud guard: row-count of `.npy` (or already-loaded ndarray) must match
    `len(load_subset(subset_path))`. Catches stale-cache + partial-write bugs that
    silently feed downstream m06/m08b the wrong data (incident 2026-04-26 — the
    9297-row eval_10k cache survived an ultra_hard_3066_eval (308) re-run because
    no module verified the subset axis).

    Args:
        arr_or_path: numpy ndarray OR path to .npy file. Path form mmaps the file
            so a 60-MB .npy doesn't trigger a full read just to inspect shape[0].
        subset_path: subset JSON path passed via --subset (None → no-op).
        label: short tag for the FATAL message (e.g. "frozen embeddings",
            "augA overlap").

    On mismatch: prints FATAL with both numbers + remediation, then `sys.exit(1)`.
    On match: silent. Subset=None → silent (caller is in --FULL no-subset mode).
    """
    if subset_path is None:
        return
    import numpy as _np
    if isinstance(arr_or_path, (str, Path)):
        arr = _np.load(arr_or_path, mmap_mode="r")
    else:
        arr = arr_or_path
    actual_n = arr.shape[0]
    expected_n = len(load_subset(subset_path))
    if actual_n != expected_n:
        print(f"FATAL: {label} row-count {actual_n} != --subset clip count "
              f"{expected_n} ({subset_path}).")
        if isinstance(arr_or_path, (str, Path)):
            print(f"  Cache file: {arr_or_path}")
            print(f"  This .npy was produced by a prior run on a DIFFERENT subset.")
        else:
            print(f"  Likely cause: partial worker run or stale checkpoint.")
        print(f"  Fix: re-run upstream with --cache-policy 2 (or set "
              f"CACHE_POLICY_ALL=2 for the eval chain).")
        sys.exit(1)


def get_output_dir(subset_path: str = None, sanity: bool = False,
                   poc: bool = False) -> Path:
    """
    Return output directory based on mode — FLAG-driven, not subset-driven.

    SANITY mode (--SANITY)         → outputs/sanity/
    POC mode    (--POC)            → outputs/poc/
    FULL mode   (--FULL [+subset]) → outputs/full/

    Note: subset_path no longer forces POC routing (2026-04-20 fix). Flag name
    now matches output directory verbatim. An iter9 `--FULL --subset subset_10k.json`
    call writes to outputs/full/ as expected. Scripts that need the old
    "subset-implies-POC" behavior must now pass `--POC` explicitly (or pass
    --output-dir on the python script to override).
    """
    if sanity:
        return OUTPUTS_SANITY_DIR
    if poc:
        return OUTPUTS_POC_DIR
    return OUTPUTS_DIR


def get_module_output_dir(module_name: str, subset_path: str = None,
                          sanity: bool = False, poc: bool = False) -> Path:
    """Return per-module output directory: outputs/{mode}/{module_name}/

    Each src/m*.py gets its own subdirectory for clean separation.
    Example: get_module_output_dir("m10_sam_segment", sanity=True)
             → outputs/sanity/m10_sam_segment/
    """
    base = get_output_dir(subset_path, sanity, poc)
    out = base / module_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def add_subset_arg(parser):
    """Add --subset argument to any argparse parser (shared across m04-m08)."""
    parser.add_argument("--subset", type=str, default=None,
                        help="Path to subset JSON (e.g., data/subset_10k_local/subset_10k.json) for POC mode")


def add_local_data_arg(parser):
    """Add --local-data argument for pre-downloaded local WebDataset shards."""
    parser.add_argument("--local-data", type=str, default=None,
                        help="Local WebDataset dir (from m00d_download_subset.py). "
                             "Bypasses HF streaming for 100%% hit rate.")


# =============================================================================
# UTILITY FUNCTIONS (avoid redundancy across modules)
# =============================================================================

def check_gpu():
    """Check if CUDA GPU is available. Exit if not."""
    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU not available.")
        print("This script requires Nvidia GPU. No CPU fallback.")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")


def get_all_clips() -> list:
    """
    Get all video clips from CLIPS_DIR (recursive).
    Returns list of Path objects.
    """
    if not CLIPS_DIR.exists():
        return []
    return sorted(CLIPS_DIR.rglob("*.mp4"))


def get_all_videos() -> list:
    """
    Get all videos from VIDEOS_DIR.
    Returns list of Path objects.
    """
    return list(VIDEOS_DIR.glob("*.mp4"))


def get_video_duration(video_path) -> float:
    """Get video duration in seconds using ffprobe."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError) as e:
        # iter15 audit (2026-05-15): preserve 0.0-sentinel contract (m02 callers
        # check `dur <= 0` to skip corrupt videos) BUT print the error class +
        # path so the failure is observable, not silent (CLAUDE.md FAIL LOUD).
        print(f"  [WARN] get_video_duration({video_path}): "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 0.0


def sanitize_name(name: str) -> str:
    """Sanitize name for use as directory name (lowercase, underscores)."""
    name = name.lower().strip()
    name = re.sub(r'[,\s]+', '_', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def build_video_section_map() -> dict:
    """Build video_id → section path mapping from YT_videos_raw.json.

    Returns dict like: {"qABnYGIilHE": "tier1/bangalore/drive", ...}
    """
    if not YT_VIDEOS_JSON.exists():
        print(f"WARNING: {YT_VIDEOS_JSON} not found, cannot build section map")
        return {}

    with open(YT_VIDEOS_JSON, 'r') as f:
        data = json.load(f)

    mapping = {}

    # Drive tours (tier1)
    for city, vids in data.get("drive_tours", {}).items():
        for v in vids:
            if v.get("id"):
                mapping[v["id"]] = f"tier1/{city}/drive"

    # Drone views (tier1)
    for city, vids in data.get("drone_views", {}).items():
        for v in vids:
            if v.get("id"):
                mapping[v["id"]] = f"tier1/{city}/drone"

    # Walking tours (tier1 + goa)
    for city, vids in data.get("walking_tours", {}).items():
        for v in vids:
            if v.get("id"):
                if city == "goa":
                    mapping[v["id"]] = "goa/walking"
                else:
                    mapping[v["id"]] = f"tier1/{city}/walking"

    # Tier 2
    for city, city_data in data.get("tier2_cities", {}).items():
        for tour_type in ["drive", "walking", "drone", "rain"]:
            for v in city_data.get(tour_type, []):
                if v.get("id"):
                    mapping[v["id"]] = f"tier2/{city}/{tour_type}"

    # Monuments
    for m in data.get("monuments", []):
        monument_name = sanitize_name(m.get("name", "unknown"))
        city = m.get("city", "")
        if city:
            dir_name = f"{monument_name}_{sanitize_name(city)}"
        else:
            dir_name = monument_name
        for tour_type in ["walking_tours", "drive_tours", "drone_views"]:
            for v in m.get(tour_type, []):
                if v.get("id"):
                    mapping[v["id"]] = f"monuments/{dir_name}"

    return mapping


# Processed-video tracking (for m02 with clip-count verification)
PROCESSED_VIDEOS_FILE = CLIPS_DIR / ".processed.json"


def get_processed_video_ids() -> dict:
    """Get dict of {video_id: clip_count} already processed by m02."""
    if not PROCESSED_VIDEOS_FILE.exists():
        # Migrate from old .processed.txt if it exists
        old_file = CLIPS_DIR / ".processed.txt"
        if old_file.exists():
            ids = set(old_file.read_text().strip().split('\n')) - {''}
            migrated = {vid_id: -1 for vid_id in ids}  # -1 = unknown count
            with open(PROCESSED_VIDEOS_FILE, 'w') as f:
                json.dump(migrated, f)
            old_file.unlink()
            print(f"Migrated {len(migrated)} entries from .processed.txt → .processed.json")
            return migrated
        return {}
    with open(PROCESSED_VIDEOS_FILE) as f:
        return json.load(f)


def mark_video_processed(video_id: str, clip_count: int):
    """Record that a video has been processed with its clip count."""
    PROCESSED_VIDEOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    processed = get_processed_video_ids()
    processed[video_id] = clip_count
    with open(PROCESSED_VIDEOS_FILE, 'w') as f:
        json.dump(processed, f)


def load_embeddings_and_tags() -> tuple:
    """
    Load embeddings and tags, verify alignment.
    Returns (embeddings, tags) or exits on error.
    """
    import numpy as np

    # Load embeddings
    if not EMBEDDINGS_FILE.exists():
        print(f"ERROR: Embeddings not found: {EMBEDDINGS_FILE}")
        print("Run m05_vjepa_embed.py first")
        sys.exit(1)

    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load tags
    if not TAGS_FILE.exists():
        print(f"ERROR: Tags not found: {TAGS_FILE}")
        print("Run m04_vlm_tag.py first")
        sys.exit(1)

    with open(TAGS_FILE, 'r') as f:
        tags = json.load(f)
    print(f"Loaded tags: {len(tags)} clips")

    # Verify alignment
    if len(tags) != embeddings.shape[0]:
        print(f"WARNING: Mismatch - {embeddings.shape[0]} embeddings vs {len(tags)} tags")
        min_len = min(len(tags), embeddings.shape[0])
        embeddings = embeddings[:min_len]
        tags = tags[:min_len]

    return embeddings, tags


# =============================================================================
# BATCH PROCESSING CONFIG (optimized for A100-40GB)
# =============================================================================
DEFAULT_BATCH_SIZE = get_pipeline_config()["gpu"]["default_batch_size"]
DEFAULT_NUM_WORKERS = get_pipeline_config()["gpu"]["default_num_workers"]
RAM_CACHE_DIR = Path("/dev/shm/video_clips_cache")


def setup_ram_cache(clip_paths: list, use_cache: bool = True, cache_subdir: str = "clips") -> tuple:
    """
    Copy clips to RAM (/dev/shm) for faster I/O.

    Args:
        clip_paths: List of clip paths
        use_cache: Whether to use RAM cache
        cache_subdir: Subdirectory name in /dev/shm

    Returns:
        Tuple of (cached_paths, cache_enabled)
    """
    from tqdm import tqdm

    cache_dir = Path(f"/dev/shm/{cache_subdir}_cache")

    if not use_cache:
        return clip_paths, False

    # Check available RAM in /dev/shm
    try:
        shm_stat = os.statvfs("/dev/shm")
        available_gb = (shm_stat.f_bavail * shm_stat.f_frsize) / (1024**3)

        # Estimate clip size (assume ~2MB per clip average)
        estimated_size_gb = len(clip_paths) * 2 / 1024

        if available_gb < estimated_size_gb + 2:  # Keep 2GB buffer
            print(f"RAM cache: Skipping (need {estimated_size_gb:.1f}GB, have {available_gb:.1f}GB)")
            return clip_paths, False

    except Exception as e:
        print(f"RAM cache: Skipping ({e})")
        return clip_paths, False

    print("\n=== Setting up RAM cache ===")
    print(f"Copying {len(clip_paths)} clips to /dev/shm (~{estimated_size_gb:.1f}GB)")

    # Clean up old cache
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_paths = []
    for src_path in tqdm(clip_paths, desc="Caching to RAM", unit="clip"):
        src_path = Path(src_path)
        # Use relative path from CLIPS_DIR with __ separator for hierarchical dirs
        try:
            rel = src_path.relative_to(CLIPS_DIR)
            cache_name = str(rel).replace(os.sep, "__")
        except ValueError:
            cache_name = f"{src_path.parent.name}__{src_path.name}"
        dst_path = cache_dir / cache_name
        shutil.copy2(src_path, dst_path)
        cached_paths.append(dst_path)

    print(f"RAM cache ready: {cache_dir}")
    return cached_paths, True


def cleanup_ram_cache(cache_subdir: str = "clips"):
    """Remove RAM cache after processing."""
    cache_dir = Path(f"/dev/shm/{cache_subdir}_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("RAM cache cleaned up")


def get_deduplicated_clips() -> list:
    """
    Get clip paths from m04's deduplicated output (embeddings.paths.npy).
    Falls back to all clips if embeddings not available.

    Returns:
        List of Path objects for clips
    """
    import numpy as np

    paths_file = EMBEDDINGS_FILE.with_suffix('.paths.npy')

    if paths_file.exists():
        clip_paths = np.load(paths_file, allow_pickle=True).tolist()
        print(f"Loaded {len(clip_paths)} deduplicated clips from {paths_file.name}")
        return [Path(p) for p in clip_paths]
    else:
        print(f"WARNING: {paths_file.name} not found. Run m05_vjepa_embed.py first.")
        print("Falling back to all clips (may cause misalignment with embeddings)")
        return get_all_clips()


def restore_original_path(cache_path: Path, clips_dir: Path = None) -> str:
    """Convert RAM cache path back to original path (supports hierarchical dirs)."""
    if clips_dir is None:
        clips_dir = CLIPS_DIR
    cache_name = cache_path.name
    # Replace __ back to path separators (hierarchical structure)
    rel_path = cache_name.replace("__", os.sep)
    restored = clips_dir / rel_path
    if restored.exists():
        return str(restored)
    # Fallback: old-style single __ split
    parts = cache_name.split("__", 1)
    if len(parts) == 2:
        return str(clips_dir / parts[0] / parts[1])
    return str(cache_path)


def check_output_exists(output_paths: list, description: str = "output") -> bool:
    """
    Check if output files exist and prompt user for action.

    Args:
        output_paths: List of Path objects to check
        description: Description of the output for display

    Returns:
        True if should process (delete and re-run)
        False if should skip (use cached files)
    """
    # Convert to Path objects and check existence
    existing = []
    for p in output_paths:
        p = Path(p)
        if p.exists():
            existing.append(p)
        elif p.is_dir() or (p.parent.exists() and any(p.parent.iterdir())):
            # Check if it's a directory pattern
            if p.is_dir():
                existing.append(p)

    if not existing:
        return True  # No existing files, proceed with processing

    # Display existing files
    print(f"\n{'='*50}")
    print(f"Found existing {description}:")
    for p in existing[:5]:  # Show max 5
        if p.is_dir():
            count = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {p}/ ({count} files)")
        else:
            print(f"  {p}")
    if len(existing) > 5:
        print(f"  ... and {len(existing) - 5} more")
    print(f"{'='*50}")

    # Auto-use cached (no interactive prompt — use `rm` to force re-run)
    # See shell script docstring for cache control commands
    print("Using cached files, skipping processing.")
    return False


# ═════════════════════════════════════════════════════════════════════════
# CLI — shell scripts call these instead of inline python -c "..."
# ═════════════════════════════════════════════════════════════════════════

def _get_nested(d: dict, key_path: str):
    """Traverse nested dict by dot-separated key path. e.g. 'optimization.max_epochs.full'."""
    for key in key_path.split("."):
        d = d[key]
    return d


if __name__ == "__main__":
    usage = """Usage:
  python -u src/utils/config.py get-yaml <yaml_path> <key_path>
  python -u src/utils/config.py get-json <json_path> <key>

Examples:
  python -u src/utils/config.py get-yaml configs/pretrain/vitg16_indian.yaml optimization.max_epochs.full
  python -u src/utils/config.py get-yaml configs/pipeline.yaml verify.sanity_min_clips
  python -u src/utils/config.py get-json outputs/full/ablation_winner.json winner_lambda
  python -u src/utils/config.py get-json outputs/full/m09_lambda0_001/training_summary.json epochs
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "get-yaml":
        if len(sys.argv) != 4:
            print("Usage: get-yaml <yaml_path> <key_path>")
            sys.exit(1)
        import yaml
        with open(sys.argv[2]) as f:
            cfg = yaml.safe_load(f)
        print(_get_nested(cfg, sys.argv[3]))

    elif cmd == "get-json":
        if len(sys.argv) != 4:
            print("Usage: get-json <json_path> <key>")
            sys.exit(1)
        with open(sys.argv[2]) as f:
            data = json.load(f)
        print(data[sys.argv[3]])

    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
