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
    write_action_labels_json,
)
from utils.eval_subset import stratified_by_motion_class_subset


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
      cfg["probe_action_labels"]["eval_subset_in"][mode_dir]      → source eval pool
      cfg["probe_action_labels"]["poc_subset_out"]                → POC stratified subset path
      cfg["probe_action_labels"]["min_clips_per_class"][mode_dir] → label filter floor
      cfg["probe_action_labels"]["min_per_split"][mode_dir]       → split floor
      cfg["probe_action_labels"]["n_motion_classes"]              → POC target_per_class divisor
      cfg["data"]["poc_total_clips"]                              → POC clip budget
      cfg["probe_taxonomy_labels"]["tags_json"]                   → taxonomy source
      cfg["probe_taxonomy_labels"]["tag_taxonomy"]                → taxonomy schema
      cfg["probe_taxonomy_labels"]["local_data"]                  → m04d output dir

    Args:
        mode_flag:        argparse mode (--SANITY | --POC | --FULL).
        project_root:     repo root (Path); paths in cfg resolved relative to it.
        cache_policy:     int or str, forwarded to probe_taxonomy.py subprocess.
        cfg:              merged yaml config dict.
        motion_features:  optional override for the motion_features.npy path.
                          Defaults to <local_data>/m04d_motion_features/motion_features.npy.

    Returns:
        dict: {action_path, taxonomy_path, action_generated, taxonomy_generated,
               taxonomy_skipped_reason, n_classes, n_records}.

    Raises:
        ValueError: unknown mode_flag.
        FileNotFoundError: missing required input (eval_subset, motion_features, etc.).
        subprocess.CalledProcessError: probe_taxonomy.py exits non-zero.
    """
    mode_dir = _mode_dir_from_flag(mode_flag)
    project_root = Path(project_root).resolve()

    # ─── All paths/numbers from cfg (no module-level constants) ─────────
    pal = cfg["probe_action_labels"]                 # FAIL LOUD on missing
    ptl = cfg["probe_taxonomy_labels"]
    eval_subset = project_root / pal["eval_subset_in"][mode_dir]
    min_clips_per_class = pal["min_clips_per_class"][mode_dir]
    min_per_split       = pal["min_per_split"][mode_dir]

    local_data = project_root / ptl["local_data"]
    if motion_features is None:
        motion_features = local_data / "m04d_motion_features" / "motion_features.npy"
    else:
        motion_features = Path(motion_features)

    output_action_dir   = project_root / "outputs" / mode_dir / "probe_action"
    output_taxonomy_dir = project_root / "outputs" / mode_dir / "probe_taxonomy"
    action_path   = output_action_dir / "action_labels.json"
    taxonomy_path = output_taxonomy_dir / "taxonomy_labels.json"

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

        # iter14 recipe-v3: POC stratified-by-motion-class subsampling, in-process.
        # SANITY + FULL skip this step (use the source pool directly).
        labels_input = eval_subset
        if mode_dir == "poc":
            poc_subset_out = project_root / pal["poc_subset_out"]
            n_motion_classes = pal["n_motion_classes"]
            poc_total_clips  = cfg["data"]["poc_total_clips"]
            target_per_class = max(1, poc_total_clips // n_motion_classes)
            poc_stale = (
                not poc_subset_out.exists()
                or eval_subset.stat().st_mtime > poc_subset_out.stat().st_mtime
                or motion_features.stat().st_mtime > poc_subset_out.stat().st_mtime
            )
            if poc_stale:
                print(
                    f"  [probe_labels] POC stratified-by-motion-class subset → "
                    f"{poc_subset_out}  (target_per_class={target_per_class}, "
                    f"n_motion_classes={n_motion_classes})"
                )
                src = json.loads(eval_subset.read_text())
                out = stratified_by_motion_class_subset(
                    src, motion_features, target_per_class
                )
                out["source"] = (
                    f"stratified_by_motion_class_{target_per_class}_per_class_"
                    f"of_{eval_subset.name}"
                )
                poc_subset_out.parent.mkdir(parents=True, exist_ok=True)
                poc_subset_out.write_text(json.dumps(out, indent=2))
            else:
                print(f"  [probe_labels] POC subset fresh: {poc_subset_out}")
            labels_input = poc_subset_out

        print(
            f"  [probe_labels] generating {action_path}  "
            f"(eval_subset={labels_input.name}, "
            f"min_clips_per_class={min_clips_per_class}, min_per_split={min_per_split})"
        )
        records, class_names = load_subset_with_labels(
            labels_input, motion_features, min_clips_per_class=min_clips_per_class
        )
        splits = stratified_split(records, seed=99, min_per_split=min_per_split)
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

    tags_json = project_root / ptl["tags_json"]
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

    # Taxonomy stage still uses subprocess for parity with run_eval.sh.
    print(
        f"  [probe_labels] missing: {taxonomy_path} — auto-generating "
        f"via probe_taxonomy.py --stage labels (CPU, ~30 s)"
    )
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
