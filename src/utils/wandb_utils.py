"""Shared wandb helpers. All functions are no-ops when run is None (--no-wandb)."""
import argparse


def add_wandb_args(parser: argparse.ArgumentParser):
    """Add --no-wandb flag to any argparse parser."""
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")


def init_wandb(module_name: str, mode: str, config: dict = None,
               enabled: bool = True):
    """
    Init wandb run. Returns run object or None if disabled/failed.

    Args:
        module_name: e.g. "m04", "m05", "m06", "m07"
        mode: e.g. "SANITY", "FULL", "POC"
        config: dict of run config (args, etc.)
        enabled: False when --no-wandb is passed
    """
    if not enabled:
        return None
    try:
        import wandb
        run = wandb.init(
            project="walkindia-200k",
            name=f"{module_name}_{mode}",
            config=config or {},
            reinit=True,
        )
        print(f"wandb: run={run.name} url={run.url}")
        return run
    except Exception as e:
        print(f"wandb: init failed ({e}), continuing without wandb")
        return None


def log_metrics(run, metrics: dict, step: int = None):
    """Log metrics dict if run is active."""
    if run is None:
        return
    try:
        if step is not None:
            run.log(metrics, step=step)
        else:
            run.log(metrics)
    except Exception:
        pass


def log_image(run, key: str, path: str):
    """Log image file as wandb.Image."""
    if run is None:
        return
    try:
        import wandb
        from pathlib import Path
        p = Path(path)
        if p.exists():
            run.log({key: wandb.Image(str(p))})
    except Exception:
        pass


def log_artifact(run, name: str, path: str, artifact_type: str = "output"):
    """Log file as wandb artifact."""
    if run is None:
        return
    try:
        import wandb
        from pathlib import Path
        p = Path(path)
        if p.exists():
            art = wandb.Artifact(name, type=artifact_type)
            art.add_file(str(p))
            run.log_artifact(art)
    except Exception:
        pass


def finish_wandb(run):
    """Close wandb run."""
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
