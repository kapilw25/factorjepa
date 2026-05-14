"""Shared CLI + config-merge + probe-pipeline helpers for m09a / m09c trainers.

iter13 v13 (2026-05-07): R1+R2+R5 utils refactor — extracts duplicated
boilerplate from m09a1_pretrain_encoder.py and m09c1_surgery_encoder.py into reusable
functions. Each trainer's main() and merge_config_with_args calls these
helpers + adds its own technique-specific args/logic.

Public API:
    add_m09_common_args(parser)      — argparse builder for shared CLI args
    merge_m09_common_config(cfg, args, mode_key)
                                     — flatten probe block + memory-saver flags
                                       + delegate to merge_multi_task_config /
                                       merge_motion_aux_config; defensive
                                       (isinstance check supports scalar overrides)
    setup_probe_pipeline(cfg, args, output_dir)
                                     — build_probe_clips + load_action_labels
                                       with --probe-action-labels CLI override.
                                       Returns (probe_clips, probe_labels).
"""
import sys
from pathlib import Path

from utils.config import (
    add_subset_arg, add_local_data_arg,
    add_model_config_arg, add_train_config_arg,
)
from utils.cache_policy import add_cache_policy_arg
from utils.multi_task_loss import merge_multi_task_config
from utils.motion_aux_loss import merge_motion_aux_config


# ─────────────────────────────────────────────────────────────────────────
# R1 — add_m09_common_args(parser)
# ─────────────────────────────────────────────────────────────────────────

def add_m09_common_args(parser, *, require_val_data: bool = False) -> None:
    """Bundle 16 shared CLI args used by both m09a1_pretrain_encoder + m09c1_surgery_encoder.

    Caller adds technique-specific args AFTER (e.g. m09c adds --factor-dir,
    --factor-streaming; m09a adds nothing extra).

    Args:
      require_val_data: If True, --val-subset and --val-local-data are
        registered as `required=True` (m09a contract — pretrain MUST have
        external val data). If False (default), they are optional with
        default=None (m09c contract — surgery can use train_val_split).

    Args registered (all kwargs):
      --SANITY / --POC / --FULL     mode flags
      --config                       legacy single yaml
      --model-config / --train-config  via shared adders
      --batch-size / --max-epochs / --output-dir
      --lambda-reg                   drift-control λ override
      --val-subset / --val-local-data  val data (required=require_val_data)
      --probe-subset / --probe-local-data / --probe-tags / --probe-action-labels
      --no-probe                     force-disable mid-training probe
      --taxonomy-labels-json / --no-multi-task
      --motion-features-path / --no-motion-aux
      --subset / --local-data        via shared adders
      --cache-policy                 via shared adder

    NOT included (technique-specific):
      m09c: --factor-dir, --factor-streaming, --no-factor-streaming
    """
    parser.add_argument("--config", type=str, default=None,
                        help="Legacy single YAML (back-compat).")
    add_model_config_arg(parser)
    add_train_config_arg(parser)
    parser.add_argument("--SANITY", action="store_true",
                        help="Quick validation (24 GB code-path correctness).")
    parser.add_argument("--POC", action="store_true",
                        help="POC subset (~10K clips).")
    parser.add_argument("--FULL", action="store_true",
                        help="Full training run.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config.")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max epochs from config.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory.")
    parser.add_argument("--lambda-reg", type=float, default=None,
                        help="Override drift_control.lambda_reg from CLI. "
                             "Setting --lambda-reg 0 also flips drift_control.enabled=False.")
    if require_val_data:
        parser.add_argument("--val-subset", required=True,
                            help="Path to val subset JSON (also threaded into cfg.data.val_subset).")
        parser.add_argument("--val-local-data", required=True,
                            help="Local WebDataset dir for val clips (also threaded into cfg.data.val_local_data).")
    else:
        parser.add_argument("--val-subset", type=str, default=None,
                            help="Path to val subset JSON (overrides cfg.data.val_subset).")
        parser.add_argument("--val-local-data", type=str, default=None,
                            help="Local WebDataset dir for val clips (overrides cfg.data.val_local_data).")
    # Mid-training probe block (top-1 + motion-cos + future-L1 trio).
    parser.add_argument("--probe-subset", type=str, default=None,
                        help="Path to probe-eval subset JSON (overrides cfg.probe.subset).")
    parser.add_argument("--probe-local-data", type=str, default=None,
                        help="Local WebDataset dir for probe clips (overrides cfg.probe.local_data).")
    parser.add_argument("--probe-tags", type=str, default=None,
                        help="Path to tags.json for probe clips (overrides cfg.probe.tags_path).")
    parser.add_argument("--probe-action-labels", type=str, default=None,
                        help="Path to action_labels.json (default: derive from "
                             "outputs/<mode>/probe_action/action_labels.json).")
    parser.add_argument("--no-probe", action="store_true",
                        help="Force-disable mid-training probe (overrides cfg.probe.enabled).")
    # Multi-task probe-head supervision (16-dim taxonomy CE+BCE).
    parser.add_argument("--taxonomy-labels-json", type=str, default=None,
                        help="Path to taxonomy_labels.json (overrides cfg.multi_task_probe.labels_path). "
                             "Produced by `probe_taxonomy.py --stage labels`.")
    parser.add_argument("--no-multi-task", action="store_true",
                        help="Force-disable multi-task probe-head supervision "
                             "(overrides cfg.multi_task_probe.enabled).")
    # Motion_aux loss (joint K-class CE + 13-D MSE on RAFT-flow features).
    parser.add_argument("--motion-features-path", type=Path, default=None,
                        help="Path to m04d motion_features.npy (overrides "
                             "cfg.motion_aux.motion_features_path).")
    parser.add_argument("--no-motion-aux", action="store_true",
                        help="Force-disable motion_aux supervised loss "
                             "(overrides cfg.motion_aux.enabled).")
    # Subset / local-data / cache-policy via shared adders.
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_cache_policy_arg(parser)


# ─────────────────────────────────────────────────────────────────────────
# R2 — merge_m09_common_config(cfg, args, mode_key)
# ─────────────────────────────────────────────────────────────────────────

def merge_m09_common_config(cfg: dict, args, mode_key: str) -> None:
    """Apply CLI overrides + per-mode flatten of dict-shaped yaml fields.

    Mutates `cfg` in place. Caller has already loaded yaml + set mode_key.
    Replaces ~80 LoC of duplicated boilerplate across m09a + m09c.

    Steps:
      1. data overrides: subset, local_data, val_subset, val_local_data
      2. optimization overrides: max_epochs (per-mode flatten if dict),
         batch_size, memory-saver flags (use_8bit_optim / gradient_checkpointing /
         paged_optim — defensive isinstance check supports scalar overrides)
      3. drift_control: --lambda-reg override + auto-disable when λ=0
      4. probe block: per-mode flatten of all gate-style booleans (enabled,
         best_ckpt_enabled, kill_switch_enabled, plateau_enabled,
         prec_plateau_enabled, bwt_trigger_enabled, use_permanent_val);
         CLI path overrides for probe-subset/local-data/tags
      5. delegate to merge_multi_task_config + merge_motion_aux_config

    Does NOT touch:
      - data.train_val_split (m09c-specific, surgery-internal split)
      - factor_streaming (m09c-specific)
      - layer_freeze (m09a-specific static freeze)
      - surgery.stages (m09c-specific)
    """
    # 1) data overrides
    if getattr(args, "subset", None):
        cfg["data"]["subset"] = args.subset
    if getattr(args, "local_data", None):
        cfg["data"]["local_data"] = args.local_data
    if getattr(args, "val_subset", None):
        cfg["data"]["val_subset"] = args.val_subset
    if getattr(args, "val_local_data", None):
        cfg["data"]["val_local_data"] = args.val_local_data

    # 2) optimization overrides
    # Per-mode flatten: max_epochs may be a dict {sanity, poc, full} OR scalar.
    me = cfg["optimization"]["max_epochs"]
    if isinstance(me, dict):
        cfg["optimization"]["max_epochs"] = me[mode_key]
    if getattr(args, "batch_size", None) is not None:
        cfg["optimization"]["batch_size"] = args.batch_size
    if getattr(args, "max_epochs", None) is not None:
        cfg["optimization"]["max_epochs"] = args.max_epochs
    # Defensive flatten of memory-saver flags (D4-fix: handle scalar overrides).
    for k in ("use_8bit_optim", "gradient_checkpointing", "paged_optim"):
        if k in cfg["optimization"] and isinstance(cfg["optimization"][k], dict):
            cfg["optimization"][k] = cfg["optimization"][k][mode_key]

    # 3) drift_control: --lambda-reg CLI + auto-disable on λ=0
    if getattr(args, "lambda_reg", None) is not None:
        cfg["drift_control"]["lambda_reg"] = args.lambda_reg
        if args.lambda_reg == 0:
            cfg["drift_control"]["enabled"] = False

    # 4) probe block — per-mode flatten + CLI path overrides
    if "probe" in cfg:
        probe_cfg = cfg["probe"]
        if getattr(args, "probe_subset", None):
            probe_cfg["subset"] = args.probe_subset
        if getattr(args, "probe_local_data", None):
            probe_cfg["local_data"] = args.probe_local_data
        if getattr(args, "probe_tags", None):
            probe_cfg["tags_path"] = args.probe_tags
        # Per-mode flatten of all gate-style booleans.
        for k in ("enabled", "best_ckpt_enabled", "kill_switch_enabled",
                  "plateau_enabled", "prec_plateau_enabled",
                  "bwt_trigger_enabled", "use_permanent_val"):
            if k in probe_cfg and isinstance(probe_cfg[k], dict):
                probe_cfg[k] = probe_cfg[k][mode_key]
        if getattr(args, "no_probe", False):
            probe_cfg["enabled"] = False

    # 5) Delegate to multi_task + motion_aux per-mode flatten + CLI overrides
    merge_multi_task_config(cfg, args, mode_key)
    merge_motion_aux_config(cfg, args, mode_key)


# ─────────────────────────────────────────────────────────────────────────
# R5 — setup_probe_pipeline(cfg, args, output_dir)
# ─────────────────────────────────────────────────────────────────────────

def setup_probe_pipeline(cfg: dict, args, output_dir, *,
                         subset_keys_override=None):
    """Build probe_clips + load_action_labels.

    Returns: (probe_clips, probe_labels). Either may be None when probe is
    disabled (SANITY default) OR when action_labels.json is missing.

    Action-labels resolution order (D5-fix):
      1. --probe-action-labels CLI arg
      2. derived from outputs/<mode>/probe_action/action_labels.json

    Args:
      subset_keys_override: optional set[str] of clip_keys overriding probe.subset
        (used by m09c to feed in-stage held-out val_keys instead of external val_1k).
    """
    # Lazy imports — utils.training has heavy deps (torch, faiss); only pay
    # them when the caller actually needs the probe pipeline.
    from utils.training import build_probe_clips
    from utils.action_labels import load_action_labels

    # Mode token for default action_labels path.
    if getattr(args, "SANITY", False):
        mode_subdir = "sanity"
    elif getattr(args, "POC", False):
        mode_subdir = "poc"
    else:
        mode_subdir = "full"

    # iter14 recipe-v2 (2026-05-09): FAIL LOUD per CLAUDE.md "no DEFAULT".
    # Missing probe block → OK (some yamls don't configure probe at all).
    # Present probe block → MUST have explicit `enabled: true|false` key;
    # `.get("enabled", False)` would silently treat a typo'd / missing key as disabled.
    probe_cfg = cfg["probe"] if "probe" in cfg else None
    if probe_cfg is None:
        return None, None
    if "enabled" not in probe_cfg:
        print("❌ FATAL [probe]: cfg['probe'] block present but missing 'enabled' key. "
              "Set `enabled: true` or `enabled: false` explicitly in yaml.", file=sys.stderr)
        sys.exit(3)
    if not probe_cfg["enabled"]:
        return None, None

    # Resolve probe data paths (CLI > yaml).
    subset_path = getattr(args, "probe_subset", None) or probe_cfg.get("subset")
    local_data_path = getattr(args, "probe_local_data", None) or probe_cfg.get("local_data")
    tags_path = getattr(args, "probe_tags", None) or probe_cfg.get("tags_path")
    if not subset_path or not local_data_path:
        return None, None

    # FAIL LOUD on missing keys (CLAUDE.md): cfg["data"][...] not cfg.get(...).
    num_frames = cfg["data"]["num_frames"]
    crop_size = cfg["data"]["crop_size"]
    max_clips = cfg["monitoring"]["knn_probe_clips"]

    probe_clips = build_probe_clips(
        probe_subset_path=subset_path,
        probe_local_data=local_data_path,
        probe_tags_path=tags_path,
        num_frames=num_frames, crop_size=crop_size,
        subset_keys_override=subset_keys_override,
        max_clips=max_clips,
    )

    # Action labels path: CLI > derived default.
    if getattr(args, "probe_action_labels", None):
        action_labels_path = Path(args.probe_action_labels)
    else:
        action_labels_path = (Path(__file__).resolve().parent.parent.parent /
                              f"outputs/{mode_subdir}/probe_action/action_labels.json")

    probe_labels = (load_action_labels(action_labels_path)
                    if action_labels_path.exists() else None)

    return probe_clips, probe_labels
