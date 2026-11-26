"""Hardcode auditor — static AST/regex sweep enforcing src/CLAUDE.md "No DEFAULT,
no hardcoded paths, no FALLBACK".

Rule classes (each maps to a CLAUDE.md clause):
  H1  path-literal in .py        — str constant containing data/|outputs/|configs/|
                                   checkpoints/|logs/ or ending in a data extension
                                   (.json/.pt/.npy/.npz/.csv/.tar/.yaml/.mp4).
                                   Docstrings + argparse help= strings are exempt
                                   (documentation, not consumed values).
  H2  argparse default=          — add_argument(..., default=<literal != None>).
  H3  .get(key, default)         — 2-arg .get() with a literal default (silent fallback).
  H4  getattr(args, k, default)  — 3-arg getattr on `args` (errors_N_fixes #79).
  H5  module-level path constant — top-level Assign whose value is Path("...") or a
                                   path-like string.
  H6  fn-signature literal default — def f(x="data/...", n=1000): path-like strings
                                   or magic numbers (not None/bool/0/1/-1) in defaults.
  H7  .sh inline path            — non-comment line with a repo data path that is not
                                   $VAR-expanded / yaml_extract-derived on that line.

Severity (every exemption is DECLARED here, policy-as-code — nothing silently dropped):
  must-fix — a CONSUMED value: opened/joined/compared/configured from a literal.
  review   — (a) H1 literal whose nearest enclosing call is display-only
                 (print / sys.exit / logging / exception-constructor f-strings):
                 documentation of a path, not consumption of it;
             (b) H3 .get() on an ACCUMULATOR receiver (counter/grouping dicts) —
                 the universal `d.get(k, 0) + 1` idiom, not a cfg fallback.
             H3 on cfg-class receivers (cfg/args/pr/record/summary/best_state/
             *_cfg/*_block) is ALWAYS must-fix.

Magic-number COMPARISONS are delegated to ruff PLR2004 (installed) — run alongside:
  venv_denseworld/bin/ruff check --select PLR2004 src/

Gold-standard references (2026-06-06 websearch):
  - Semgrep custom policy rules (industry standard, Python+Bash): https://semgrep.dev/docs/writing-rules/rule-syntax
  - ruff PLR2004 magic-value-comparison (pylint R2004 port): https://docs.astral.sh/ruff/rules/magic-value-comparison/
This module is the zero-new-dependency implementation of the same policy-as-code
idea, specialized to THIS repo's rules (semgrep would need the same custom rules
authored in YAML; deps are managed only via setup_env_uv.sh).

USAGE:
    # full audit (text report to stdout; exit 1 if findings and --strict):
    python -u src/utils/hardcode_audit.py \
        --py-globs "src/*.py" "src/utils/*.py" \
        --sh-globs "scripts/*.sh" \
        --exclude legacy \
        2>&1 | tee logs/hardcode_audit_$(date +%Y%m%d_%H%M%S).log

    # machine-readable:
    python -u src/utils/hardcode_audit.py --py-globs "src/*.py" --sh-globs "scripts/*.sh" \
        --exclude legacy --json outputs/audit/hardcode_audit.json
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import re
import sys
from pathlib import Path

# iter18 W7 (PLR2004, self-applied): pattern arities.
_MIN_PATHISH_LEN = 5   # shorter strings can't be meaningful paths
_GET_ARITY = 2         # d.get(key, default)
_GETATTR_ARITY = 3     # getattr(obj, name, default)

# Path-ish detection: repo data roots OR data-file extensions.
_PATH_ROOT_RE = re.compile(r"(^|[\s\"'=(])(data|outputs|configs|checkpoints|logs)/[^\s\"']*")
_PATH_EXT_RE = re.compile(r"\.(json|pt|pth|npy|npz|csv|tar|ya?ml|mp4|safetensors)$")
# .sh line is exempt when the path is built from a variable / yaml_extract / tee log target.
_SH_EXEMPT_RE = re.compile(r"\$\{?[A-Za-z_]|yaml_extract|^\s*#|^\s*$")
_SH_PATH_RE = re.compile(r"(^|[\s\"'=])(data|outputs|configs|checkpoints)/[^\s\"';]+")

# Numbers that are not "magic" in signature defaults.
_BENIGN_NUMS = {0, 1, -1, True, False}

# Display-only call roots: a path literal inside these is documentation, not consumption.
_DISPLAY_FUNCS = {"print", "exit", "log_metrics", "warn", "warning", "info", "debug",
                  "error", "write"}
# cfg-class receiver name fragments: .get() defaults on these are ALWAYS must-fix.
_CFG_RECV_RE = re.compile(r"(cfg|config|args|^pr$|record|summary|best_state|_block|"
                           r"probe|monitoring|opt|drift|surgery|stage|info)", re.I)


def _is_pathish(s: str) -> bool:
    if not isinstance(s, str) or len(s) < _MIN_PATHISH_LEN or " " in s.strip():
        return bool(isinstance(s, str) and _PATH_ROOT_RE.search(s or ""))
    return bool(_PATH_ROOT_RE.search(s) or _PATH_EXT_RE.search(s))


class _PyAuditor(ast.NodeVisitor):
    def __init__(self, path: str, src: str):
        self.path = path
        self.findings: list[dict] = []
        self._doc_lines: set[int] = set()
        self._help_lines: set[int] = set()
        self._display_spans: list[tuple[int, int]] = []
        self._src_lines = src.splitlines()
        tree = ast.parse(src)
        self.tree = tree
        # Display-only call spans (print/log/exit/raise) — H1 inside = review.
        for node in ast.walk(tree):
            fn_name = None
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    fn_name = node.func.attr
                if fn_name in _DISPLAY_FUNCS:
                    self._display_spans.append((node.lineno, node.end_lineno or node.lineno))
            if isinstance(node, ast.Raise) and node.exc is not None:
                self._display_spans.append((node.lineno, node.end_lineno or node.lineno))
        # Docstring line spans (module/class/function first-Expr constants).
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    d = body[0].value
                    self._doc_lines.update(range(d.lineno, (d.end_lineno or d.lineno) + 1))
        # argparse help= string spans (documentation, not consumed).
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "add_argument":
                for kw in node.keywords:
                    if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                        v = kw.value
                        self._help_lines.update(range(v.lineno, (v.end_lineno or v.lineno) + 1))

    def _add(self, rule: str, node: ast.AST, snippet: str, severity: str = "must-fix"):
        # Inline pragma (noqa-style, USE SPARINGLY): a line ending `# audit-ok: <reason>`
        # downgrades to review. Legitimate ONLY for external-API schemas where the
        # upstream contract makes fields optional (e.g. yt-dlp metadata) — never
        # for our own cfg/artifacts.
        ln = getattr(node, "lineno", 0)
        if "# audit-ok" in self._src_lines[ln - 1] if 0 < ln <= len(self._src_lines) else False:
            severity = "review"
        self.findings.append({"rule": rule, "file": self.path,
                              "line": ln,
                              "severity": severity,
                              "snippet": snippet[:120]})

    def _in_display(self, node: ast.AST) -> bool:
        ln = getattr(node, "lineno", 0)
        return any(a <= ln <= b for a, b in self._display_spans)

    # H1 — consumed path literals (docstrings + help= exempt; display calls = review).
    # DECLARED fragment exemptions (derivation logic, not configuration) → review:
    #   leading '.', '*' or '-'  : extension / glob fragments (".tmp.npz", "*.mp4", "-*.tar")
    #   '_tmp.' / '.tmp.' infix  : atomic-write temp suffixes (process-internal, never
    #                              cross-module; the os.replace target carries the real name)
    #   multi-line strings       : README / model-card / usage TEMPLATES (documentation
    #                              artifacts, not consumed paths)
    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str) and _is_pathish(node.value) \
                and node.lineno not in self._doc_lines \
                and node.lineno not in self._help_lines:
            s = node.value
            multiline = getattr(node, "end_lineno", node.lineno) != node.lineno or "\n" in s
            frag = "/" not in s and (s[:1] in ".*-" or s.startswith("_tmp.")
                                     or ".tmp." in s)
            sev = ("review" if (self._in_display(node) or frag or multiline)
                   else "must-fix")
            self._add("H1-path-literal", node, repr(s), sev)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # H2 — argparse default=<literal != None>. DECLARED exemption (review,
        # not must-fix): identity elements {0, ''} = "feature off / from-start /
        # no-nesting" — they inject no configuration. True/False and any other
        # literal remain must-fix (default=True silently enables behavior).
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant) \
                        and kw.value.value is not None:
                    v = kw.value.value
                    sev = ("review" if (not isinstance(v, bool) and v in (0, ""))
                           else "must-fix")
                    self._add("H2-argparse-default", node,
                              f"add_argument(..., default={v!r})", sev)
        # H3 — 2-arg .get() with literal default; cfg-class receiver = must-fix,
        # accumulator receiver = review (the `d.get(k, 0) + 1` idiom).
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and len(node.args) == _GET_ARITY and isinstance(node.args[1], ast.Constant):
            recv = ast.unparse(node.func.value)[:40]
            sev = "must-fix" if _CFG_RECV_RE.search(recv) else "review"
            self._add("H3-get-default", node,
                      f"{recv}.get(..., {node.args[1].value!r})", sev)
        # H4 — getattr(args, "k", default)
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" \
                and len(node.args) == _GETATTR_ARITY and isinstance(node.args[0], ast.Name) \
                and node.args[0].id == "args":
            self._add("H4-getattr-args-default", node, ast.unparse(node)[:100])
        self.generic_visit(node)

    # H5 — module-level path constants
    def _check_module_assigns(self):
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                v = node.value
                if isinstance(v, ast.Constant) and isinstance(v.value, str) and _is_pathish(v.value):
                    self._add("H5-module-path-const", node, ast.unparse(node)[:100])
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "Path" \
                        and v.args and isinstance(v.args[0], ast.Constant):
                    self._add("H5-module-path-const", node, ast.unparse(node)[:100])

    # H6 — function-signature literal defaults (path strings + magic numbers)
    def visit_FunctionDef(self, node: ast.FunctionDef):
        for d in list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]:
            if isinstance(d, ast.Constant):
                val = d.value
                if isinstance(val, str) and _is_pathish(val):
                    self._add("H6-signature-path-default", node,
                              f"def {node.name}(... ={val!r})")
                elif isinstance(val, (int, float)) and not isinstance(val, bool) \
                        and val not in _BENIGN_NUMS:
                    self._add("H6-signature-magic-default", node,
                              f"def {node.name}(... ={val!r})")
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def run(self) -> list[dict]:
        self._check_module_assigns()
        self.visit(self.tree)
        return self.findings


def audit_py(path: str) -> list[dict]:
    src = Path(path).read_text(encoding="utf-8", errors="replace")
    return _PyAuditor(path, src).run()


def audit_sh(path: str) -> list[dict]:
    findings = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if _SH_EXEMPT_RE.search(line) or "# audit-ok" in line:
            continue
        m = _SH_PATH_RE.search(line)
        if m:
            findings.append({"rule": "H7-sh-inline-path", "file": path, "line": i,
                             "snippet": line.strip()[:120]})
    return findings


def run_audit(py_globs: list[str], sh_globs: list[str], exclude: list[str]) -> list[dict]:
    """Importable entry point. Returns the full findings list."""
    findings: list[dict] = []
    seen: set = set()
    for g in py_globs:
        for f in sorted(glob.glob(g)):
            if f in seen or any(x in f for x in exclude):
                continue
            seen.add(f)
            findings += audit_py(f)
    for g in sh_globs:
        for f in sorted(glob.glob(g)):
            if f in seen or any(x in f for x in exclude):
                continue
            seen.add(f)
            findings += audit_sh(f)
    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--py-globs", nargs="+", required=True,
                    help="Globs of .py files to audit (e.g. 'src/*.py' 'src/utils/*.py').")
    ap.add_argument("--sh-globs", nargs="+", required=True,
                    help="Globs of .sh files to audit (e.g. 'scripts/*.sh').")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Path substrings to skip (e.g. legacy).")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Optional output JSON path for machine-readable findings.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 when findings exist (CI gate mode).")
    args = ap.parse_args()

    findings = run_audit(args.py_globs, args.sh_globs, args.exclude)

    by_rule: dict[str, list[dict]] = {}
    for f in findings:
        by_rule.setdefault(f["rule"], []).append(f)

    n_must = sum(1 for f in findings if f.get("severity", "must-fix") == "must-fix")
    print(f"\n═══ hardcode_audit · {len(findings)} findings "
          f"({n_must} must-fix · {len(findings) - n_must} review) across "
          f"{len({f['file'] for f in findings})} files ═══")
    for rule in sorted(by_rule):
        items = by_rule[rule]
        nm = sum(1 for i in items if i.get("severity", "must-fix") == "must-fix")
        print(f"\n── {rule} · {len(items)} ({nm} must-fix) ──")
        for it in items:
            tag = "❗" if it.get("severity", "must-fix") == "must-fix" else "·"
            print(f"  {tag} {it['file']}:{it['line']}: {it['snippet']}")
    by_file: dict[str, int] = {}
    for f in findings:
        by_file[f["file"]] = by_file.get(f["file"], 0) + 1
    print("\n── per-file totals ──")
    for f, n in sorted(by_file.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {f}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(findings, indent=1))
        print(f"\nwrote {args.json_out}")

    if args.strict and findings:
        sys.exit(1)


if __name__ == "__main__":
    main()
