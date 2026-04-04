"""
Output guard: verify existing outputs before expensive GPU work.

Two modes:
1. Per-script: verify_or_skip() called at top of main() in each GPU script.
2. Pipeline preflight: preflight_pipeline() called once at start of shell script,
   checks ALL steps' inputs/outputs, prints summary table, asks user to confirm.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np


def verify_or_skip(
    output_dir: Path,
    required_files: dict,
    min_clips: int = 0,
    label: str = "",
) -> bool:
    """Check if outputs exist and are valid. Skip expensive GPU work if so.

    Args:
        output_dir: Directory containing output files.
        required_files: Dict of {description: Path} to check.
            For .npy files, also checks shape.
            For .json files, checks it's valid JSON.
        min_clips: Minimum expected clip count (0 = don't check).
        label: Display label for logging (e.g., "m05 V-JEPA embed").

    Returns:
        True if all outputs valid → caller should return early (skip).
        False if any output missing/invalid → caller should proceed.
    """
    if not required_files:
        return False

    prefix = f"[output_guard] {label}: " if label else "[output_guard] "
    all_ok = True
    clip_count = None
    details = []

    for desc, path in required_files.items():
        path = Path(path)
        if not path.exists():
            details.append(f"  MISS  {desc}: {path.name}")
            all_ok = False
            continue

        # Validate by file type
        if path.suffix == ".npy":
            try:
                arr = np.load(path, allow_pickle=True)
                if hasattr(arr, 'shape') and len(arr.shape) >= 1:
                    n = arr.shape[0]
                    shape_str = f"{arr.shape}"
                    details.append(f"  OK    {desc}: {shape_str}")
                    if clip_count is None:
                        clip_count = n
                    elif n != clip_count:
                        details.append(f"  FAIL  {desc}: clip count {n} != {clip_count} (mismatch)")
                        all_ok = False
                else:
                    details.append(f"  OK    {desc}: scalar/object")
            except Exception as e:
                details.append(f"  FAIL  {desc}: corrupt ({e})")
                all_ok = False

        elif path.suffix == ".json":
            try:
                data = json.load(open(path))
                if isinstance(data, dict):
                    keys = len(data)
                    details.append(f"  OK    {desc}: {keys} keys")
                elif isinstance(data, list):
                    details.append(f"  OK    {desc}: {len(data)} items")
                else:
                    details.append(f"  OK    {desc}: loaded")
            except Exception as e:
                details.append(f"  FAIL  {desc}: corrupt ({e})")
                all_ok = False

        elif path.suffix == ".pt":
            size_mb = path.stat().st_size / 1e6
            details.append(f"  OK    {desc}: {size_mb:.0f} MB")

        elif path.suffix in (".png", ".pdf", ".csv", ".tex"):
            size_kb = path.stat().st_size / 1e3
            details.append(f"  OK    {desc}: {size_kb:.0f} KB")

        else:
            details.append(f"  OK    {desc}: exists")

    # Check minimum clip count
    if min_clips > 0 and clip_count is not None and clip_count < min_clips:
        details.append(f"  FAIL  clip count {clip_count} < min {min_clips}")
        all_ok = False

    # If files missing, attempt HF download before giving up
    if not all_ok:
        missing_files = [Path(p) for desc, p in required_files.items() if not Path(p).exists()]
        if missing_files:
            try:
                from utils.hf_outputs import download_outputs
                print(f"{prefix}attempting HF download for {len(missing_files)} missing file(s)...")
                if download_outputs(str(output_dir)):
                    # Re-check after download
                    still_missing = [f for f in missing_files if not f.exists()]
                    if not still_missing:
                        print(f"{prefix}HF download recovered all missing files")
                        return verify_or_skip(output_dir, required_files, min_clips, label)
                    else:
                        print(f"{prefix}{len(still_missing)} file(s) still missing after HF download")
            except ImportError:
                print(f"{prefix}hf_outputs not available, skipping HF download")
            except Exception as e:
                print(f"{prefix}HF download failed: {e}")

    # Print summary
    if all_ok:
        print(f"{prefix}ALL VALID — skipping (cached)")
        for d in details:
            print(d)
        return True
    else:
        print(f"{prefix}outputs missing or invalid — will re-compute")
        for d in details:
            print(d)
        return False


def verify_training_output(output_dir: Path, min_epochs: float = 0) -> bool:
    """Check if m09 training output is valid and has sufficient epochs.

    Args:
        output_dir: Lambda output directory (e.g., outputs/poc/m09_lambda0_001/).
        min_epochs: Minimum epoch count required. 0 = any completed training.

    Returns:
        True if student_encoder.pt exists with sufficient epochs → skip.
        False if missing or insufficient → re-train.
    """
    student = output_dir / "student_encoder.pt"
    summary = output_dir / "training_summary.json"

    if not student.exists():
        print(f"[output_guard] student_encoder.pt not found — will train")
        return False

    if not summary.exists():
        print(f"[output_guard] training_summary.json not found — will train")
        return False

    try:
        s = json.load(open(summary))
        epochs = s.get("epochs", 0)
        loss = s.get("final_jepa_loss", "?")
        steps = s.get("steps", "?")
        size_mb = student.stat().st_size / 1e6

        if min_epochs > 0 and epochs < min_epochs:
            print(f"[output_guard] student has {epochs} epochs, need {min_epochs} — will re-train")
            return False

        print(f"[output_guard] Training output valid — skipping")
        print(f"  OK    student_encoder.pt: {size_mb:.0f} MB")
        print(f"  OK    epochs={epochs}, steps={steps}, jepa_loss={loss}")
        return True

    except Exception as e:
        print(f"[output_guard] training_summary.json corrupt ({e}) — will train")
        return False


# ═════════════════════════════════════════════════════════════════════════
# PIPELINE PREFLIGHT — called once at start of shell script
# ═════════════════════════════════════════════════════════════════════════

def _check_file(path: Path) -> str:
    """Quick status string for a file."""
    if not path.exists():
        return "MISS"
    if path.suffix == ".npy":
        try:
            arr = np.load(path, allow_pickle=True)
            return f"OK ({arr.shape})" if hasattr(arr, 'shape') else "OK"
        except Exception:
            return "CORRUPT"
    elif path.suffix == ".json":
        try:
            json.load(open(path))
            return "OK"
        except Exception:
            return "CORRUPT"
    elif path.suffix == ".pt":
        return f"OK ({path.stat().st_size / 1e6:.0f}MB)"
    else:
        return "OK"


def preflight_pipeline(steps: list, interactive: bool = True) -> dict:
    """Check all pipeline steps' inputs/outputs before GPU work starts.

    Args:
        steps: List of dicts, each with:
            - name: Step display name (e.g., "m09 train lambda=0")
            - inputs: Dict of {desc: Path} — required inputs
            - outputs: Dict of {desc: Path} — expected outputs
            - est_time: Estimated time string (e.g., "~33 min")
        interactive: If True, prompt user to confirm. If False, auto-proceed.

    Returns:
        Dict with:
            - will_run: List of step names that need to run (outputs missing)
            - will_skip: List of step names that will be skipped (outputs exist)
            - missing_inputs: List of (step_name, input_desc) for missing inputs
            - proceed: Bool — True if user confirmed (or non-interactive)
    """
    will_run = []
    will_skip = []
    missing_inputs = []

    print("\n" + "=" * 70)
    print("  PIPELINE PREFLIGHT — checking all inputs/outputs before GPU work")
    print("=" * 70)

    for step in steps:
        name = step["name"]
        inputs = step.get("inputs", {})
        outputs = step.get("outputs", {})
        est = step.get("est_time", "")

        # Check inputs
        input_ok = True
        for desc, path in inputs.items():
            status = _check_file(Path(path))
            if status == "MISS" or status == "CORRUPT":
                missing_inputs.append((name, desc, str(path)))
                input_ok = False

        # Check outputs
        all_outputs_exist = True
        for desc, path in outputs.items():
            if not Path(path).exists():
                all_outputs_exist = False
                break

        if all_outputs_exist and outputs:
            will_skip.append(name)
            status_str = "\033[32mSKIP (cached)\033[0m"
        elif not input_ok:
            will_run.append(name)
            status_str = "\033[31mBLOCKED (missing input)\033[0m"
        else:
            will_run.append(name)
            status_str = f"\033[33mWILL RUN\033[0m {est}"

        print(f"  {status_str:45s}  {name}")

    print()
    print(f"  Steps to run:  {len(will_run)}")
    print(f"  Steps cached:  {len(will_skip)}")

    if missing_inputs:
        print(f"\n  \033[31mMISSING INPUTS ({len(missing_inputs)}):\033[0m")
        for step_name, desc, path in missing_inputs:
            print(f"    {step_name} needs {desc}: {path}")

    print("=" * 70)

    proceed = True
    if interactive and (will_run or missing_inputs):
        print("\nOptions:")
        print("  [1] Proceed (run steps marked WILL RUN, skip cached)")
        print("  [2] Abort")
        try:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice != "1":
                print("Aborted by user.")
                proceed = False
        except (EOFError, KeyboardInterrupt):
            # Non-interactive (piped input) — auto-proceed
            proceed = True

    return {
        "will_run": will_run,
        "will_skip": will_skip,
        "missing_inputs": missing_inputs,
        "proceed": proceed,
    }


# ═════════════════════════════════════════════════════════════════════════
# CLI — called from shell scripts: python -u src/utils/output_guard.py preflight_pretrain ...
# ═════════════════════════════════════════════════════════════════════════

def _build_pretrain_steps(out_dir: str, lambdas: list, winner_encoder: str,
                          shuffled_encoder: str, config_path: str) -> list:
    """Build step list for run_pretrain.sh preflight."""
    import yaml
    cfg = yaml.safe_load(open(config_path))
    winner_epochs = cfg["optimization"]["max_epochs"].get("winner", 5)
    out = Path(out_dir)
    steps = []

    # Phase 1: Lambda ablation training
    for lam_val, lam_dir in lambdas:
        lam_out = out / f"m09_{lam_dir}"
        steps.append({
            "name": f"m09 train (λ={lam_val})",
            "inputs": {},  # pretrained weights downloaded on-demand
            "outputs": {
                "student": str(lam_out / "student_encoder.pt"),
                "summary": str(lam_out / "training_summary.json"),
            },
            "est_time": "~33 min",
        })

    # Phase 2: Winner deep train + embed + metrics
    winner_out = out / f"m09_{winner_encoder.replace('vjepa_', '')}"
    steps.append({
        "name": f"m09 deep train ({winner_encoder}, {winner_epochs}ep)",
        "inputs": {},
        "outputs": {
            "student": str(winner_out / "student_encoder.pt"),
        },
        "est_time": f"~{winner_epochs * 33} min",
    })
    steps.append({
        "name": f"m05 re-embed ({winner_encoder})",
        "inputs": {"student": str(winner_out / "student_encoder.pt")},
        "outputs": {
            "embeddings": str(out / f"embeddings_{winner_encoder}.npy"),
            "paths": str(out / f"embeddings_{winner_encoder}.paths.npy"),
        },
        "est_time": "~1h 47min",
    })
    steps.append({
        "name": f"m06 metrics ({winner_encoder})",
        "inputs": {
            "embeddings": str(out / f"embeddings_{winner_encoder}.npy"),
            "tags": str(out / "tags.json"),
        },
        "outputs": {
            "metrics": str(out / f"m06_metrics_{winner_encoder}.json"),
        },
        "est_time": "~1 min",
    })

    # Phase 3: Temporal, shuffled, UMAP, plots, compare
    steps.append({
        "name": f"m06b temporal ({winner_encoder})",
        "inputs": {
            "embeddings": str(out / f"embeddings_{winner_encoder}.npy"),
            "motion": str(out / "motion_features.npy"),
        },
        "outputs": {
            "temporal": str(out / f"m06b_temporal_corr_{winner_encoder}.json"),
        },
        "est_time": "~2 min",
    })
    steps.append({
        "name": f"m05 shuffled ({shuffled_encoder})",
        "inputs": {"student": str(winner_out / "student_encoder.pt")},
        "outputs": {
            "embeddings": str(out / f"embeddings_{shuffled_encoder}.npy"),
        },
        "est_time": "~1h 47min",
    })
    steps.append({
        "name": f"m07 UMAP ({winner_encoder})",
        "inputs": {
            "embeddings": str(out / f"embeddings_{winner_encoder}.npy"),
        },
        "outputs": {
            "umap": str(out / f"umap_2d_{winner_encoder}.npy"),
        },
        "est_time": "~5 min",
    })
    steps.append({
        "name": "m08b compare (all encoders)",
        "inputs": {
            "frozen_metrics": str(out / "m06_metrics.json"),
            "adapted_metrics": str(out / f"m06_metrics_{winner_encoder}.json"),
        },
        "outputs": {
            "radar": str(out / "m08b_radar.png"),
        },
        "est_time": "~30 sec",
    })

    return steps


def _build_evaluate_steps(out_dir: str, tags_file: str, local_data: str) -> list:
    """Build step list for run_evaluate.sh preflight."""
    out = Path(out_dir)
    encoders = ["vjepa", "random", "dinov2", "clip", "vjepa_shuffled"]
    steps = []

    # m00d: pre-download
    steps.append({
        "name": "m00d pre-download subset",
        "inputs": {},
        "outputs": {"manifest": str(Path(local_data) / "manifest.json")} if local_data else {},
        "est_time": "~39 min",
    })

    # m04: VLM tagging
    steps.append({
        "name": "m04 VLM tagging (Qwen)",
        "inputs": {},
        "outputs": {"tags": tags_file},
        "est_time": "~3h",
    })

    # m05: V-JEPA embeddings
    steps.append({
        "name": "m05 V-JEPA embeddings",
        "inputs": {},
        "outputs": {
            "embeddings": str(out / "embeddings.npy"),
            "paths": str(out / "embeddings.paths.npy"),
        },
        "est_time": "~1h 47min",
    })

    # m05b: Baselines (4 encoders)
    for enc in ["random", "dinov2", "clip", "vjepa_shuffled"]:
        sfx = {"random": "_random", "dinov2": "_dinov2",
               "clip": "_clip", "vjepa_shuffled": "_vjepa_shuffled"}[enc]
        steps.append({
            "name": f"m05b {enc} embeddings",
            "inputs": {},
            "outputs": {
                "embeddings": str(out / f"embeddings{sfx}.npy"),
                "paths": str(out / f"embeddings{sfx}.paths.npy"),
            },
            "est_time": "~30 min" if enc != "random" else "~1 sec",
        })

    # m05c: True Overlap augmented embeddings
    steps.append({
        "name": "m05c True Overlap augmented",
        "inputs": {"embeddings": str(out / "embeddings.npy")},
        "outputs": {
            "augA": str(out / "overlap_augA.npy"),
            "augB": str(out / "overlap_augB.npy"),
        },
        "est_time": "~1h",
    })

    # m04d: Motion features
    steps.append({
        "name": "m04d motion features",
        "inputs": {},
        "outputs": {
            "motion": str(out / "motion_features.npy"),
            "motion_paths": str(out / "motion_features.paths.npy"),
        },
        "est_time": "~40 min",
    })

    # m06: FAISS metrics (per encoder)
    for enc in encoders:
        sfx = {"vjepa": "", "random": "_random", "dinov2": "_dinov2",
               "clip": "_clip", "vjepa_shuffled": "_vjepa_shuffled"}[enc]
        steps.append({
            "name": f"m06 metrics ({enc})",
            "inputs": {
                "embeddings": str(out / f"embeddings{sfx}.npy"),
                "tags": tags_file,
            },
            "outputs": {
                "metrics": str(out / f"m06_metrics{sfx}.json"),
                "knn": str(out / f"knn_indices{sfx}.npy"),
            },
            "est_time": "~1 min",
        })

    # m06b: Temporal correlation (per encoder)
    for enc in encoders:
        sfx = {"vjepa": "", "random": "_random", "dinov2": "_dinov2",
               "clip": "_clip", "vjepa_shuffled": "_vjepa_shuffled"}[enc]
        steps.append({
            "name": f"m06b temporal ({enc})",
            "inputs": {
                "embeddings": str(out / f"embeddings{sfx}.npy"),
                "motion": str(out / "motion_features.npy"),
            },
            "outputs": {
                "temporal": str(out / f"m06b_temporal_corr{sfx}.json"),
            },
            "est_time": "~2 min",
        })

    # m07: UMAP (per encoder)
    for enc in encoders:
        sfx = {"vjepa": "", "random": "_random", "dinov2": "_dinov2",
               "clip": "_clip", "vjepa_shuffled": "_vjepa_shuffled"}[enc]
        steps.append({
            "name": f"m07 UMAP ({enc})",
            "inputs": {"embeddings": str(out / f"embeddings{sfx}.npy")},
            "outputs": {"umap": str(out / f"umap_2d{sfx}.npy")},
            "est_time": "~3 min",
        })

    # m08b: Compare
    steps.append({
        "name": "m08b compare (all encoders)",
        "inputs": {"frozen_metrics": str(out / "m06_metrics.json")},
        "outputs": {"radar": str(out / "m08b_radar.png")},
        "est_time": "~30 sec",
    })

    return steps


def preflight_gpu_packages(pipeline: str, config: str = "", out_dir: str = ""):
    """Check GPU + packages before any GPU work. Shared between run_evaluate.sh and run_pretrain.sh.

    USAGE:
        python -u src/utils/output_guard.py preflight_gpu pretrain configs/pretrain/vitg16_indian.yaml outputs/full
        python -u src/utils/output_guard.py preflight_gpu evaluate
    """
    errors = []

    # GPU + torch
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA not available")
        else:
            props = torch.cuda.get_device_properties(0)
            print(f"GPU:            {torch.cuda.get_device_name(0)}, VRAM: {props.total_memory/1e9:.0f}GB")
            print(f"PyTorch:        {torch.__version__}")
            print(f"CUDA:           {torch.version.cuda}")
    except ImportError:
        errors.append("torch not installed")

    # FAISS-GPU
    try:
        import faiss
        if faiss.get_num_gpus() == 0:
            errors.append("FAISS-GPU: 0 GPUs detected (need >= 1)")
        else:
            print(f"FAISS GPUs:     {faiss.get_num_gpus()}")
    except ImportError:
        errors.append("faiss not installed")

    # Flash-Attention 2
    try:
        import flash_attn
        print(f"Flash-Attn:     {flash_attn.__version__}")
    except ImportError:
        errors.append("flash_attn not installed (V-JEPA requires FA2)")

    # cuML (m07 UMAP)
    try:
        import cuml
        print(f"cuML:           {cuml.__version__}")
    except ImportError:
        errors.append("cuml not installed (m07 UMAP needs cuML)")

    # transformers
    try:
        import transformers
        print(f"Transformers:   {transformers.__version__}")
    except ImportError:
        errors.append("transformers not installed")

    # wandb (optional)
    try:
        import wandb
        print(f"wandb:          {wandb.__version__}")
    except ImportError:
        print("wandb:          NOT INSTALLED (ok, using --no-wandb)")

    # HF Token
    try:
        from dotenv import load_dotenv
        load_dotenv()
        t = os.getenv("HF_TOKEN")
        if t:
            print(f"HF_TOKEN:       {t[:10]}...")
        else:
            errors.append("HF_TOKEN not found in .env (private dataset needs auth)")
    except ImportError:
        errors.append("python-dotenv not installed")

    # Pipeline-specific checks
    if pipeline == "pretrain":
        # vjepa2 dependency
        if not os.path.exists("deps/vjepa2/src/models/vision_transformer.py"):
            errors.append("deps/vjepa2 not found. Run: git clone --depth 1 https://github.com/facebookresearch/vjepa2.git deps/vjepa2")
        else:
            print("vjepa2:         deps/vjepa2/ present")

        # Config file
        if config and not os.path.exists(config):
            errors.append(f"Config not found: {config}")
        elif config:
            print(f"Config:         {config}")

        # Ch9 baseline (needed for m08b only, not training)
        if out_dir:
            baseline = os.path.join(out_dir, "m06_metrics.json")
            if os.path.exists(baseline):
                m = json.load(open(baseline))
                print(f"Baseline:       Prec@K={m['easy']['prec_at_k']:.1f}% (Ch9 frozen)")
            else:
                print(f"Baseline:       NOT FOUND ({baseline}) — m08b will fail, training will proceed")
                print(f"  Fix after Ch9: python -u src/utils/hf_outputs.py download outputs/full")

    if errors:
        print(f"\nFATAL: {len(errors)} check(s) failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nPre-flight: ALL PASSED")


def verify_training_artifacts(output_dir: str):
    """Check training outputs exist and print summary. Called after m09 training step.

    USAGE:
        python -u src/utils/output_guard.py verify_training <output_dir>
    """
    out = output_dir
    for f in ["student_encoder.pt", "training_summary.json", "loss_log.csv"]:
        path = os.path.join(out, f)
        if os.path.exists(path):
            if f.endswith(".json"):
                s = json.load(open(path))
                print(f"  OK   {f:30s} jepa_loss={s['final_jepa_loss']:.4f}  epochs={s['epochs']}")
            elif f.endswith(".pt"):
                size_mb = os.path.getsize(path) / 1e6
                print(f"  OK   {f:30s} {size_mb:.0f} MB")
            else:
                lines = sum(1 for _ in open(path)) - 1
                print(f"  OK   {f:30s} {lines} steps")
        else:
            print(f"  MISS {f}")


def verify_pretrain_final(output_dir: str, config_path: str):
    """Final verification of all Ch10 outputs. Reads lambdas from YAML config.

    USAGE:
        python -u src/utils/output_guard.py verify_pretrain_final outputs/full configs/pretrain/vitg16_indian.yaml
    """
    import yaml

    out = output_dir
    ok = 0
    fail = 0

    def check(label, path):
        nonlocal ok, fail
        if os.path.exists(path):
            print(f"  OK   {label}")
            ok += 1
        else:
            print(f"  MISS {label}")
            fail += 1

    # Read lambdas from config (not hardcoded)
    cfg = yaml.safe_load(open(config_path))
    ablation_lambdas = cfg["drift_control"]["ablation_lambdas"]
    lambdas = []
    for lam in ablation_lambdas:
        lam_str = f"{lam:g}".replace(".", "_")
        lambdas.append((f"{lam:g}", f"lambda{lam_str}"))

    print("=== ABLATION TRAINING (Phase 1) ===")
    print(f"  {'lambda':>10s} {'JEPA Loss':>12s} {'Steps':>8s} {'Student':>10s}")
    print(f"  " + "-" * 45)
    for lam_val, lam_dir in lambdas:
        lam_out = os.path.join(out, f"m09_{lam_dir}")
        spath = os.path.join(lam_out, "training_summary.json")
        if os.path.exists(spath):
            s = json.load(open(spath))
            has_student = "OK" if os.path.exists(os.path.join(lam_out, "student_encoder.pt")) else "MISS"
            print(f"  {lam_val:>10s} {s['final_jepa_loss']:>12.4f} {s['steps']:>8d} {has_student:>10s}")
            ok += 1
        else:
            print(f"  {lam_val:>10s} MISSING")
            fail += 1

    # Winner info
    winner_path = os.path.join(out, "ablation_winner.json")
    winner_dir = None
    if os.path.exists(winner_path):
        w = json.load(open(winner_path))
        winner_dir = w["winner_dir"]
        val_loss = w.get("winner_val_loss", "?")
        print(f"  WINNER: lambda={w['winner_lambda']} (val_loss={val_loss})")
        ok += 1

    print()
    print("=== WINNER RUN (Phase 2) ===")
    if winner_dir:
        enc = f"vjepa_{winner_dir}"
        check("Winner embeddings", os.path.join(out, f"embeddings_{enc}.npy"))
        check("Winner metrics", os.path.join(out, f"m06_metrics_{enc}.json"))
        check("Winner knn_indices", os.path.join(out, f"knn_indices_{enc}.npy"))

    print()
    print("=== FULL EVALUATION (Phase 3) ===")
    if winner_dir:
        enc = f"vjepa_{winner_dir}"
        check("Winner UMAP", os.path.join(out, f"umap_2d_{enc}.npy"))
        check("Winner UMAP plot", os.path.join(out, f"m08_umap_{enc}.png"))
        check("Comparison radar", os.path.join(out, "m08b_radar.png"))
        check("Comparison table", os.path.join(out, "m08b_comparison_table.tex"))

    print()
    print(f"=== TOTAL: {ok} OK, {fail} MISSING ===")


def verify_evaluate_final(output_dir: str, tags_file: str):
    """Final verification of all Ch9 outputs (tags, 5 encoders, overlap, metrics, UMAP, plots).

    USAGE:
        python -u src/utils/output_guard.py verify_evaluate_final outputs/full outputs/full/tags.json
    """
    out = output_dir
    ok_count = 0
    fail_count = 0

    def check(label, path, validator=None):
        nonlocal ok_count, fail_count
        if os.path.exists(path):
            detail = ""
            if validator:
                detail = validator(path)
            print(f"  OK   {label:40s} {detail}")
            ok_count += 1
        else:
            print(f"  MISS {label:40s} {path}")
            fail_count += 1

    # Tags
    print("=== TAGS (v3 taxonomy) ===")
    check("tags.json", tags_file,
          lambda p: f"{len(json.load(open(p)))} clips, {len(json.load(open(p))[0].keys())} fields")

    # Embeddings (5 encoders)
    print()
    print("=== EMBEDDINGS (5 encoders) ===")
    for enc, sfx, dim in [("vjepa", "", 1408), ("random", "_random", 1408), ("dinov2", "_dinov2", 1536),
                           ("clip", "_clip", 768), ("vjepa_shuffled", "_vjepa_shuffled", 1408)]:
        check(f"{enc} embeddings", os.path.join(out, f"embeddings{sfx}.npy"),
              lambda p: str(np.load(p).shape))
        check(f"{enc} paths", os.path.join(out, f"embeddings{sfx}.paths.npy"),
              lambda p: f"{len(np.load(p, allow_pickle=True))} keys")

    # True Overlap
    print()
    print("=== TRUE OVERLAP ===")
    check("overlap_augA.npy", os.path.join(out, "overlap_augA.npy"),
          lambda p: str(np.load(p).shape))
    check("overlap_augB.npy", os.path.join(out, "overlap_augB.npy"),
          lambda p: str(np.load(p).shape))

    # Metrics (5 encoders)
    print()
    print("=== METRICS (5 encoders) ===")
    print(f"  {'Encoder':20s} {'Prec@K':>8s} {'mAP@K':>8s} {'Cycle@K':>8s} {'nDCG@K':>8s}")
    print("  " + "-" * 58)
    for enc, sfx in [("vjepa", ""), ("random", "_random"), ("dinov2", "_dinov2"),
                      ("clip", "_clip"), ("vjepa_shuffled", "_vjepa_shuffled")]:
        mpath = os.path.join(out, f"m06_metrics{sfx}.json")
        if os.path.exists(mpath):
            m = json.load(open(mpath))
            e = m["easy"]
            print(f"  {enc:20s} {e['prec_at_k']:7.1f}% {e['map_at_k']:8.4f} {e['cycle_at_k']:7.1f}% {e['ndcg_at_k']:8.4f}")
            ok_count += 1
        else:
            print(f"  {enc:20s} MISSING")
            fail_count += 1

    # UMAP (5 encoders)
    print()
    print("=== UMAP (5 encoders) ===")
    for enc, sfx in [("vjepa", ""), ("random", "_random"), ("dinov2", "_dinov2"),
                      ("clip", "_clip"), ("vjepa_shuffled", "_vjepa_shuffled")]:
        check(f"{enc} umap", os.path.join(out, f"umap_2d{sfx}.npy"),
              lambda p: str(np.load(p).shape))

    # Motion features
    print()
    print("=== MOTION FEATURES (m04d) ===")
    check("motion_features.npy", os.path.join(out, "motion_features.npy"),
          lambda p: str(np.load(p).shape))

    # Temporal metrics
    print()
    print("=== TEMPORAL METRICS (m06b) ===")
    for enc, sfx in [("vjepa", ""), ("random", "_random"), ("dinov2", "_dinov2"),
                      ("clip", "_clip"), ("vjepa_shuffled", "_vjepa_shuffled")]:
        check(f"{enc} temporal", os.path.join(out, f"m06b_temporal_corr{sfx}.json"))

    # Plots
    print()
    print("=== PLOTS ===")
    for f in ["m08_umap.png", "m08_confusion_matrix.png", "m08_knn_grid.png",
              "m08b_encoder_comparison.png", "m08b_radar.png", "m08b_comparison_table.tex",
              "m08b_spatial_temporal_bar.png", "m08b_tradeoff_scatter.png"]:
        check(f"plot: {f}", os.path.join(out, f))

    print()
    print(f"=== TOTAL: {ok_count} OK, {fail_count} MISSING ===")


if __name__ == "__main__":
    usage = """Usage:
  python -u src/utils/output_guard.py preflight_gpu <pretrain|evaluate> [config] [out_dir]
  python -u src/utils/output_guard.py preflight_pretrain <out_dir> <config>
  python -u src/utils/output_guard.py preflight_evaluate <out_dir> <tags_file> <local_data>
  # Add --interactive to restore the confirmation prompt (default: auto-proceed)
  python -u src/utils/output_guard.py verify_training <output_dir>
  python -u src/utils/output_guard.py verify_pretrain_final <output_dir> <config>
  python -u src/utils/output_guard.py verify_evaluate_final <output_dir> <tags_file>
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]
    # Auto-proceed: preflight shows the plan (WILL RUN/SKIP/BLOCKED) for visibility,
    # but never blocks. Use --interactive to restore the prompt if needed.
    interactive = "--interactive" in sys.argv

    if cmd == "preflight_gpu":
        pipeline = sys.argv[2] if len(sys.argv) > 2 else "evaluate"
        config = sys.argv[3] if len(sys.argv) > 3 else ""
        out_dir = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else ""
        preflight_gpu_packages(pipeline, config, out_dir)

    elif cmd == "preflight_pretrain":
        out_dir = sys.argv[2]
        config_path = sys.argv[3]

        # Read lambdas from config (not hardcoded)
        import yaml
        cfg = yaml.safe_load(open(config_path))
        ablation_lambdas = cfg["drift_control"]["ablation_lambdas"]
        lambdas = []
        for lam in ablation_lambdas:
            lam_str = f"{lam:g}".replace(".", "_")
            lambdas.append((f"{lam:g}", f"lambda{lam_str}"))

        winner_json = Path(out_dir) / "ablation_winner.json"
        if winner_json.exists():
            w = json.load(open(winner_json))
            winner_dir = w["winner_dir"]
        else:
            winner_dir = "lambda0_001"

        winner_encoder = f"vjepa_{winner_dir}"
        shuffled_encoder = f"{winner_encoder}_shuffled"

        steps = _build_pretrain_steps(out_dir, lambdas, winner_encoder,
                                       shuffled_encoder, config_path)
        result = preflight_pipeline(steps, interactive=interactive)

        if not result["proceed"]:
            sys.exit(1)

    elif cmd == "preflight_evaluate":
        out_dir = sys.argv[2]
        tags_file = sys.argv[3]
        local_data = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else ""

        steps = _build_evaluate_steps(out_dir, tags_file, local_data)
        result = preflight_pipeline(steps, interactive=interactive)

        if not result["proceed"]:
            sys.exit(1)

    elif cmd == "verify_training":
        verify_training_artifacts(sys.argv[2])

    elif cmd == "verify_pretrain_final":
        verify_pretrain_final(sys.argv[2], sys.argv[3])

    elif cmd == "verify_evaluate_final":
        verify_evaluate_final(sys.argv[2], sys.argv[3])

    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
