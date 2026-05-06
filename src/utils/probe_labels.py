"""Auto-bootstrap probe label artifacts for m09a/m09c. CPU-only.

Ensures `outputs/<mode>/probe_action/action_labels.json` and
`outputs/<mode>/probe_taxonomy/taxonomy_labels.json` exist before training
starts. Mirrors run_probe_eval.sh Stage 1 — same probe_action.py /
probe_taxonomy.py CLI args, subprocess-isolated for behavioral parity.

Lets `python -u src/m09{a,c}.py ...` run end-to-end without requiring the
shell wrapper to have run Stage 1 first. No-op fast path: ~1ms when both
files already exist (just two stat()s).

iter13 v12 (2026-05-06): probe_action labels now derive from MOTION-flow
classes (RAFT optical-flow → 16 classes), NOT path-derived 3-class action.
Bootstrap CLI updated: --tags-json dropped, --motion-features required.
"""
import subprocess
import sys
from pathlib import Path


# Per-mode default subsets — matches run_probe_eval.sh and run_probe_train.sh
# convention. SANITY uses the 600-clip stratified pool (200/POV × 3); POC +
# FULL share the 9.9k FULL pool (POC is a downstream training-time subset,
# not a separate data manifest at this layer).
_MODE_TO_DIR = {
    "--SANITY": "sanity", "--sanity": "sanity",
    "--POC":    "poc",    "--poc":    "poc",
    "--FULL":   "full",   "--full":   "full",
}
_MODE_TO_EVAL_SUBSET = {
    "sanity": "data/eval_10k_sanity.json",
    "poc":    "data/eval_10k.json",
    "full":   "data/eval_10k.json",
}
# iter13 v12 (2026-05-06): motion-flow class filter floors. SANITY relaxes since
# 600 clips / 16 motion classes ≈ 37/class avg with skew → sparse classes after
# filter; FULL uses paper-grade thresholds (≥34/class → ≥5 per split @ 70/15/15).
_MODE_TO_MIN_CLIPS_PER_CLASS = {"sanity": 5,  "poc": 34, "full": 34}
_MODE_TO_MIN_PER_SPLIT       = {"sanity": 1,  "poc": 5,  "full": 5}


def ensure_probe_labels_for_mode(
    mode_flag: str,
    project_root: Path,
    cache_policy,
    *,
    tags_json: Path = None,
    tag_taxonomy: Path = None,
    eval_subset: Path = None,
    motion_features: Path = None,
    local_data: Path = None,
) -> dict:
    """Ensure action_labels.json + taxonomy_labels.json exist; subprocess-bootstrap if not.

    Returns a dict:
        {action_path: Path, taxonomy_path: Path | None,
         action_generated: bool, taxonomy_generated: bool,
         taxonomy_skipped_reason: str | None}

    Single source of truth for the path convention (mirrors run_probe_eval.sh).
    All optional kwargs are caller-side overrides. Defaults are derived from
    `mode_flag` + `project_root` so m09a / m09c don't duplicate the layout.

    iter13 v12 (Phase 2): motion_features.npy at <local_data>/motion_features.npy
    is REQUIRED for action labels (16-class motion-flow derivation). FAIL HARD
    if missing — caller must run m04d first.

    Raises:
      ValueError on unknown mode_flag.
      FileNotFoundError when an input dependency is missing AND a label file
        needs generation (caller can't continue with multi-task disabled
        silently — fail loud per CLAUDE.md).
      subprocess.CalledProcessError if probe_action.py / probe_taxonomy.py exit non-zero.
    """
    if mode_flag not in _MODE_TO_DIR:
        raise ValueError(
            f"ensure_probe_labels_for_mode: unknown mode_flag {mode_flag!r}; "
            f"expected one of {sorted(_MODE_TO_DIR.keys())}"
        )
    mode_dir = _MODE_TO_DIR[mode_flag]
    project_root = Path(project_root).resolve()

    if eval_subset is None:
        eval_subset = project_root / _MODE_TO_EVAL_SUBSET[mode_dir]
    else:
        eval_subset = Path(eval_subset)
    if tags_json is None:
        tags_json = project_root / "data" / "eval_10k_local" / "tags.json"
    else:
        tags_json = Path(tags_json)
    if tag_taxonomy is None:
        tag_taxonomy = project_root / "configs" / "tag_taxonomy.json"
    else:
        tag_taxonomy = Path(tag_taxonomy)
    # iter13 v12: motion_features default — alongside the dataset's TAR shards.
    if local_data is None:
        local_data = project_root / "data" / "eval_10k_local"
    else:
        local_data = Path(local_data)
    if motion_features is None:
        motion_features = local_data / "motion_features.npy"
    else:
        motion_features = Path(motion_features)
    min_clips_per_class = _MODE_TO_MIN_CLIPS_PER_CLASS[mode_dir]
    min_per_split       = _MODE_TO_MIN_PER_SPLIT[mode_dir]

    output_action_dir = project_root / "outputs" / mode_dir / "probe_action"
    output_taxonomy_dir = project_root / "outputs" / mode_dir / "probe_taxonomy"
    action_path = output_action_dir / "action_labels.json"
    taxonomy_path = output_taxonomy_dir / "taxonomy_labels.json"

    result = {
        "action_path": action_path,
        "taxonomy_path": taxonomy_path,
        "action_generated": False,
        "taxonomy_generated": False,
        "taxonomy_skipped_reason": None,
    }

    # ── 1. Action labels (motion-flow classes from m04d optical flow) ────
    if action_path.exists():
        print(f"  [probe_labels] cached: {action_path}")
    else:
        # iter13 v12: motion_features.npy is REQUIRED — no graceful disable.
        # m04d must have run first; if not, FAIL HARD with the exact command.
        for required in (eval_subset, motion_features):
            if not required.exists():
                hint = ""
                if required == motion_features:
                    hint = (
                        f"\n  Run m04d first (~67 min on Blackwell):\n"
                        f"    python -u src/m04d_motion_features.py {mode_flag} \\\n"
                        f"        --subset {eval_subset} --local-data {local_data} \\\n"
                        f"        --features-out {motion_features}\n"
                        f"  Or download via:  python -u src/utils/hf_outputs.py download-data"
                    )
                raise FileNotFoundError(
                    f"ensure_probe_labels_for_mode: cannot generate {action_path}: "
                    f"missing input {required}.{hint}"
                )
        # paths.npy companion check
        paths_companion = motion_features.with_name(motion_features.stem + ".paths.npy")
        if not paths_companion.exists():
            raise FileNotFoundError(
                f"ensure_probe_labels_for_mode: motion_features.paths.npy not found "
                f"at {paths_companion} (must be alongside .npy)"
            )
        print(f"  [probe_labels] missing: {action_path} — auto-generating "
              f"via probe_action.py --stage labels (CPU, ~1 min)")
        cmd = [
            sys.executable, "-u", str(project_root / "src" / "probe_action.py"),
            mode_flag,
            "--stage", "labels",
            "--eval-subset", str(eval_subset),
            "--motion-features", str(motion_features),
            "--min-clips-per-class", str(min_clips_per_class),
            "--min-per-split", str(min_per_split),
            "--output-root", str(output_action_dir),
            "--cache-policy", str(cache_policy),
            "--no-wandb",
        ]
        subprocess.run(cmd, check=True, cwd=str(project_root))
        result["action_generated"] = True

    # ── 2. Taxonomy labels (multi-task supervision source) ───────────────
    if taxonomy_path.exists():
        print(f"  [probe_labels] cached: {taxonomy_path}")
        return result

    if not tag_taxonomy.exists():
        reason = f"{tag_taxonomy} not found"
        result["taxonomy_skipped_reason"] = reason
        result["taxonomy_path"] = None
        print(f"  [probe_labels] WARN: cannot generate {taxonomy_path}: {reason}")
        print("    → multi_task_probe will auto-disable for this run")
        return result
    if not tags_json.exists():
        reason = f"{tags_json} not found"
        result["taxonomy_skipped_reason"] = reason
        result["taxonomy_path"] = None
        print(f"  [probe_labels] WARN: cannot generate {taxonomy_path}: {reason}")
        print("    → multi_task_probe will auto-disable for this run")
        return result

    print(f"  [probe_labels] missing: {taxonomy_path} — auto-generating "
          f"via probe_taxonomy.py --stage labels (CPU, ~30 s)")
    cmd = [
        sys.executable, "-u", str(project_root / "src" / "probe_taxonomy.py"),
        mode_flag,
        "--stage", "labels",
        "--eval-subset", str(eval_subset),
        "--tags-json", str(tags_json),
        "--tag-taxonomy", str(tag_taxonomy),
        "--output-root", str(output_taxonomy_dir),
        "--cache-policy", str(cache_policy),
    ]
    subprocess.run(cmd, check=True, cwd=str(project_root))
    result["taxonomy_generated"] = True
    return result
