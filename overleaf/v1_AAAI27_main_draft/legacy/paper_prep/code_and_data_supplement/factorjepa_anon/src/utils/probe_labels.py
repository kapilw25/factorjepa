"""Auto-bootstrap probe label artifacts for m09a/m09c. CPU-only.

Ensures `outputs/<mode>/probe_action/action_labels.json` and
`outputs/<mode>/probe_taxonomy/taxonomy_labels.json` exist before training
starts. Lets `python -u src/m09{a,c}.py ...` run end-to-end without the shell
wrapper having pre-run Stage 1. No-op fast path: ~1ms when both files already
exist (just two stat()s).

iter13 v12 (2026-05-06): probe_action labels derive from MOTION-flow classes
(RAFT optical-flow → 16 classes), NOT path-derived 3-class action.

iter14 recipe-v3 (2026-05-09):
  • POC mode now generates a stratified-by-motion-class subset
    (data/eval_10k_local/eval_10k_poc.json by default) BEFORE invoking probe_action,
    in-process via utils.eval_subset.stratified_by_motion_class_subset.
    Guarantees POC labels match FULL schema (8 motion classes after
    the 34-clip filter — CLAUDE.md POC↔FULL parity).
  • ALL paths + numbers come from cfg (probe_action_labels / probe_taxonomy_labels
    blocks in configs/train/base_optimization.yaml). No module-level constants
    per CLAUDE.md "No hardcoded values in Python".
  • Shell-side orchestration in scripts/run_train.sh + run_recipe_v3_sweep.sh
    is now redundant — m09a/m09c call this fn directly (in-process), shells stay
    thin per CLAUDE.md.
"""
import json
import subprocess
import sys
from pathlib import Path

from utils.action_labels import (
    load_subset_with_labels,
    stratified_split,
    subsample_manifest_for_mode,
    write_action_labels_json,
)
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)
# stratified_by_motion_class_subset is now called inside
# subsample_manifest_for_mode (Option X). No direct import needed here.


def _mode_dir_from_flag(mode_flag: str) -> str:
    """Normalize argparse-style mode flag (e.g. '--POC') to dir token ('poc')."""
    norm = mode_flag.lstrip("-").lower()
    if norm not in ("sanity", "poc", "full"):
        raise ValueError(
            f"_mode_dir_from_flag: unknown mode_flag {mode_flag!r}; "
            f"expected --SANITY|--POC|--FULL (any case)"
        )
    return norm


def ensure_probe_labels_for_mode(
    mode_flag: str,
    project_root: Path,
    cache_policy,
    cfg: dict,
    *,
    motion_features: Path = None,
) -> dict:
    """Ensure action_labels.json + taxonomy_labels.json exist; bootstrap in-process if not.

    Reads ALL paths + numbers from cfg (CLAUDE.md "no hardcoded values in Python"):
      cfg["data"]["local_data_dir"]                               → active data dir (iter16 M9)
      cfg["data"]["master_manifest_name"]                          → master JSON filename (iter16 M9)
      cfg["probe_action_labels"]["min_clips_per_class"][mode_dir] → label filter floor
      cfg["probe_action_labels"]["min_per_split"][mode_dir]       → split floor
      cfg["probe_action_labels"]["n_motion_classes"]              → POC target_per_class divisor
      cfg["probe_taxonomy_labels"]["tag_taxonomy"]                → taxonomy schema (configs/)

    Derived paths (iter16 M9 — flip local_data_dir to migrate):
      eval_subset = local_data_dir / master_manifest_name
      tags_json   = local_data_dir / "tags.json"
      motion_features = local_data_dir / "m04d_motion_features" / "motion_features.npy"

    Args:
        mode_flag:        argparse mode (--SANITY | --POC | --FULL).
        project_root:     repo root (Path); paths in cfg resolved relative to it.
        cache_policy:     int or str, forwarded to m04f_taxonomy_labels.py subprocess.
        cfg:              merged yaml config dict.
        motion_features:  optional override for the motion_features.npy path.
                          Defaults to <local_data>/m04d_motion_features/motion_features.npy.

    Returns:
        dict: {action_path, taxonomy_path, action_generated, taxonomy_generated,
               taxonomy_skipped_reason, n_classes, n_records}.

    Raises:
        ValueError: unknown mode_flag.
        FileNotFoundError: missing required input (eval_subset, motion_features, etc.).
        subprocess.CalledProcessError: m04f_taxonomy_labels.py exits non-zero.
    """
    mode_dir = _mode_dir_from_flag(mode_flag)
    project_root = Path(project_root).resolve()

    # ─── All paths/numbers from cfg (no module-level constants) ─────────
    pal = cfg["probe_action_labels"]                 # FAIL LOUD on missing
    ptl = cfg["probe_taxonomy_labels"]
    # iter16 M9 (2026-05-21): master manifest + local_data dir derived from
    # cfg.data.local_data_dir + cfg.data.master_manifest_name (pipeline.yaml).
    # Single source of truth — flipping the 2 yaml keys migrates the whole
    # pipeline from eval_10k_local → full_local.
    data_cfg = cfg["data"]                            # FAIL LOUD on missing
    local_data = project_root / data_cfg["local_data_dir"]
    eval_subset = local_data / data_cfg["master_manifest_name"]
    min_clips_per_class = pal["min_clips_per_class"][mode_dir]
    min_per_split       = pal["min_per_split"][mode_dir]

    if motion_features is None:
        motion_features = local_data / "m04d_motion_features" / artifact("motion_features")
    else:
        motion_features = Path(motion_features)

    output_action_dir   = project_root / "outputs" / mode_dir / artifact("probe_action_dir")
    output_taxonomy_dir = project_root / "outputs" / mode_dir / artifact("probe_taxonomy_dir")
    action_path   = output_action_dir / artifact("action_labels")
    taxonomy_path = output_taxonomy_dir / artifact("taxonomy_labels")

    result = {
        "action_path": action_path,
        "taxonomy_path": taxonomy_path,
        "action_generated": False,
        "taxonomy_generated": False,
        "taxonomy_skipped_reason": None,
        "n_classes": None,
        "n_records": None,
    }

    # ── 1. Action labels (motion-flow classes from m04d optical flow) ────
    if action_path.exists():
        print(f"  [probe_labels] cached: {action_path}")
    else:
        # Required inputs (m04d must have run first).
        for required in (eval_subset, motion_features):
            if not required.exists():
                hint = ""
                if required == motion_features:
                    hint = (
                        f"\n  Run m04d first (~67 min on Blackwell):\n"
                        f"    python -u src/m04d_motion_features.py {mode_flag} \\\n"
                        f"        --subset {eval_subset} --local-data {local_data}\n"
                        f"    (writes to {motion_features.parent}/ by default)\n"
                        f"  Or download via:  python -u src/utils/hf_outputs.py download-data"
                    )
                raise FileNotFoundError(
                    f"ensure_probe_labels_for_mode: cannot generate {action_path}: "
                    f"missing input {required}.{hint}"
                )
        paths_companion = motion_features.with_name(motion_features.stem + ".paths.npy")
        if not paths_companion.exists():
            raise FileNotFoundError(
                f"ensure_probe_labels_for_mode: motion_features.paths.npy not found "
                f"at {paths_companion} (must be alongside .npy)"
            )

        # iter16 M1 Option X (2026-05-21 PM): per-mode subsampling via the
        # shared subsample_manifest_for_mode() helper. Symmetric with
        # probe_action.run_labels_stage (CLI entry-point) so action_labels.json
        # schema is identical regardless of caller.
        #   SANITY → sorted(clip_keys)[:n]               (code check)
        #   POC    → stratified_by_motion_class_subset   (POC↔FULL parity)
        #   FULL   → identity (ratio=1.0)
        # No more per-mode pre-made eval_10k_{poc,sanity}.json on disk
        # (plan user-decision #1 — single master, runtime subsampling).
        n_motion_classes = pal["n_motion_classes"]
        src = json.loads(eval_subset.read_text())
        pool_keys = subsample_manifest_for_mode(
            mode_dir, src["clip_keys"], motion_features, n_motion_classes
        )
        print(
            f"  [probe_labels Opt X] mode={mode_dir}: subsampled "
            f"{len(pool_keys):,}/{len(src['clip_keys']):,} clips "
            f"({len(pool_keys)/max(len(src['clip_keys']),1):.2%})"
        )

        print(
            f"  [probe_labels] generating {action_path}  "
            f"(eval_subset={eval_subset.name}, "
            f"min_clips_per_class={min_clips_per_class}, min_per_split={min_per_split})"
        )
        records, class_names = load_subset_with_labels(
            eval_subset, motion_features,
            min_clips_per_class=min_clips_per_class,
            clip_keys=pool_keys,
        )
        # iter16 M1: mode-keyed probe_split (ratios in configs/pipeline.yaml).
        # Symmetric with probe_action.run_labels_stage CLI path.
        from utils.config import get_probe_split
        _split_cfg = get_probe_split(mode_dir)
        splits = stratified_split(records,
                                  train_pct=_split_cfg["train_pct"],
                                  val_pct=_split_cfg["val_pct"],
                                  seed=99,
                                  min_per_split=min_per_split,
                                  mode=mode_dir)
        output_action_dir.mkdir(parents=True, exist_ok=True)
        write_action_labels_json(records, splits, action_path)
        result["action_generated"] = True
        result["n_classes"] = len(class_names)
        result["n_records"] = len(records)
        print(
            f"  [probe_labels] wrote {action_path}: {len(records)} records · "
            f"{len(class_names)} classes"
        )

    # ── 2. Taxonomy labels (multi-task supervision source) ───────────────
    if taxonomy_path.exists():
        print(f"  [probe_labels] cached: {taxonomy_path}")
        return result

    # iter16 M9 (2026-05-21): tags_json derived from local_data (M9 yaml flip
    # migrates eval_10k_local → full_local). tag_taxonomy stays as separate
    # cfg key (lives in configs/, not in the active local data dir).
    tags_json = local_data / artifact("tags")
    tag_taxonomy = project_root / ptl["tag_taxonomy"]
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

    # Taxonomy stage still uses subprocess for parity with run_eval.sh. The old probe_taxonomy.py
    # `--stage labels` step was split out into src/m04f_taxonomy_labels.py (iter16 §3.2) — call THAT
    # (it IS the labels step, so there is no --stage; same args otherwise + --no-wandb for the CPU helper).
    print(
        f"  [probe_labels] missing: {taxonomy_path} — auto-generating "
        f"via m04f_taxonomy_labels.py (CPU, ~30 s)"
    )
    cmd = [
        sys.executable, "-u", str(project_root / "src" / "m04f_taxonomy_labels.py"),
        mode_flag,
        "--eval-subset", str(eval_subset),
        "--tags-json", str(tags_json),
        "--tag-taxonomy", str(tag_taxonomy),
        "--output-root", str(output_taxonomy_dir),
        "--cache-policy", str(cache_policy),
        "--no-wandb",
    ]
    subprocess.run(cmd, check=True, cwd=str(project_root))
    result["taxonomy_generated"] = True
    return result
