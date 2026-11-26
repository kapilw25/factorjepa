#!/usr/bin/env python3
"""Extract a dotted-key value from a yaml file. Fail-loud on missing key.

Used by scripts/run_*.sh wrappers (CLAUDE.md "thin wrapper, no inline python -c").

USAGE:
    scripts/lib/yaml_extract.py <yaml-path> <dotted.key>

Example:
    scripts/lib/yaml_extract.py configs/legacy2/explora.yaml data.module
    # → "m09b"
"""
import sys
from pathlib import Path

import yaml


def main() -> int:
    if len(sys.argv) != 3:
        print(f"USAGE: {sys.argv[0]} <yaml-path> <dotted.key>", file=sys.stderr)
        return 2

    yaml_path = Path(sys.argv[1])
    dotted_key = sys.argv[2]

    if not yaml_path.is_file():
        print(f"FATAL: yaml not found: {yaml_path}", file=sys.stderr)
        return 3

    # Walk the `extends:` chain (variant → surgery_base → base_optimization) and
    # deep-merge each parent under the child so inherited keys (data.train_subset,
    # probe.subset, etc.) resolve correctly. Mirrors src/utils/config.py:163-181.
    def _deep_merge(base: dict, overlay: dict) -> dict:
        merged = base.copy()
        for k, v in overlay.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = _deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    seen = {yaml_path.resolve()}
    cur_dir = yaml_path.parent
    while True:
        extends = cfg.pop("extends", None)
        if not extends:
            break
        parent = (cur_dir / extends).resolve()
        if parent in seen:
            print(f"FATAL: extends cycle at {parent}", file=sys.stderr)
            return 6
        seen.add(parent)
        if not parent.is_file():
            print(f"FATAL: extends target not found: {parent}", file=sys.stderr)
            return 7
        with open(parent) as f:
            parent_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(parent_cfg, cfg)
        cur_dir = parent.parent

    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            print(
                f"FATAL: key '{dotted_key}' missing from {yaml_path} (failed at '{part}')",
                file=sys.stderr,
            )
            return 4
        node = node[part]

    if isinstance(node, (dict, list)):
        print(
            f"FATAL: key '{dotted_key}' resolves to {type(node).__name__}, not scalar; "
            f"yaml_extract returns scalars only",
            file=sys.stderr,
        )
        return 5

    print(node)
    return 0


if __name__ == "__main__":
    sys.exit(main())
