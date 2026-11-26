"""AST-precise companion to hardcode_audit: rewrite artifact-name string literals
to `artifact("<key>")` lookups (pipeline.yaml `artifacts:` single source).

Safety model (why this is NOT a blind sed):
  - operates on ast.Constant nodes ONLY, located by exact (line, col, end_col)
    spans — comments can't match (not in the AST) and docstrings / argparse
    help= strings are explicitly excluded with the SAME span logic the auditor
    uses, so USAGE examples keep their readable literal names;
  - rewrites only literals whose value EXACTLY equals a pipeline.yaml
    artifacts value (reverse map value→key); everything else untouched;
  - inserts `from utils.data_paths import artifact` after the last top-level
    import if the file doesn't already import it;
  - every rewritten file must re-parse (ast.parse) before being written —
    a failed parse aborts that file with no write.

USAGE:
    # dry-run report (no writes):
    python -u src/utils/hardcode_rewrite.py --py-globs "src/*.py" "src/utils/*.py" \
        --exclude legacy --dry-run 2>&1 | tee logs/hardcode_rewrite_dryrun_$(date +%Y%m%d_%H%M%S).log
    # apply:
    python -u src/utils/hardcode_rewrite.py --py-globs "src/*.py" "src/utils/*.py" \
        --exclude legacy 2>&1 | tee logs/hardcode_rewrite_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import argparse
import ast
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import get_pipeline_config  # noqa: E402


def _protected_lines(tree: ast.AST) -> set:
    """Line spans of docstrings + argparse help= strings (auditor-identical)."""
    lines: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                d = body[0].value
                lines.update(range(d.lineno, (d.end_lineno or d.lineno) + 1))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "add_argument":
            for kw in node.keywords:
                if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                    v = kw.value
                    lines.update(range(v.lineno, (v.end_lineno or v.lineno) + 1))
    return lines


def rewrite_file(path: str, value_to_key: dict, dry_run: bool) -> int:
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    protected = _protected_lines(tree)

    # Constants living INSIDE f-strings need the {artifact('k')} field form.
    fstring_consts = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant):
                    fstring_consts.add(id(v))

    # Collect single-line replacement spans, deepest-first per line so col
    # offsets stay valid while editing right-to-left.
    spans = []   # (lineno, col, end_col, key, in_fstring)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value in value_to_key \
                and node.lineno == node.end_lineno \
                and node.lineno not in protected:
            spans.append((node.lineno, node.col_offset, node.end_col_offset,
                          value_to_key[node.value], id(node) in fstring_consts))
    if not spans:
        return 0

    lines = src.splitlines(keepends=True)
    for lineno, col, end_col, key, in_f in sorted(spans, key=lambda s: (s[0], -s[1])):
        line = lines[lineno - 1]
        repl = f"{{artifact('{key}')}}" if in_f else f'artifact("{key}")'
        lines[lineno - 1] = line[:col] + repl + line[end_col:]

    new_src = "".join(lines)

    # Ensure the import exists (skip data_paths itself + the audit/rewrite tools).
    own = Path(path).name in ("data_paths.py", "hardcode_audit.py", "hardcode_rewrite.py")
    if not own and "from utils.data_paths import" not in new_src:
        t2 = ast.parse(new_src)   # parse BEFORE import insertion to find the slot
        last_import_end = 0
        for node in t2.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last_import_end = max(last_import_end, node.end_lineno or node.lineno)
        ls = new_src.splitlines(keepends=True)
        ls.insert(last_import_end,
                  "from utils.data_paths import artifact  "
                  "# iter18 W4: canonical artifact names (pipeline.yaml)\n")
        new_src = "".join(ls)
    elif not own and "artifact" not in new_src.split("from utils.data_paths import", 1)[1].split("\n", 1)[0]:
        # data_paths imported without artifact → extend that import line.
        new_src = new_src.replace("from utils.data_paths import ",
                                  "from utils.data_paths import artifact, ", 1)

    ast.parse(new_src)   # MUST re-parse or we abort with no write
    if not dry_run:
        Path(path).write_text(new_src, encoding="utf-8")
    return len(spans)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--py-globs", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arts = get_pipeline_config()["artifacts"]
    value_to_key = {v: k for k, v in arts.items()}

    total = 0
    for g in args.py_globs:
        for f in sorted(glob.glob(g)):
            if any(x in f for x in args.exclude):
                continue
            if Path(f).name in ("hardcode_audit.py", "hardcode_rewrite.py"):
                continue
            n = rewrite_file(f, value_to_key, args.dry_run)
            if n:
                print(f"  {n:3d} rewrites  {f}")
                total += n
    print(f"\n{'DRY-RUN: would rewrite' if args.dry_run else 'rewrote'} {total} literals "
          f"({len(value_to_key)} artifact names mapped)")


if __name__ == "__main__":
    main()
