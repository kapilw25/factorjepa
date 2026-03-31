#!/bin/bash
# PostToolUse hook: Auto-run py_compile + AST structural check after Edit/Write on src/m*.py files.
# Fails (exit 1) if either check fails, forcing Claude to fix before proceeding.
# Enforces CLAUDE.md rules 6.1a (py_compile) and 6.1b (AST structure).

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Only act on Edit or Write tools
if [ "$TOOL" != "Edit" ] && [ "$TOOL" != "Write" ]; then
  exit 0
fi

# Only act on src/m*.py pipeline files
if ! echo "$FILE_PATH" | grep -qE 'src/m[0-9].*\.py$'; then
  exit 0
fi

# Extract just the filename for display
BASENAME=$(basename "$FILE_PATH")

# ── Check 1: py_compile (syntax) ─────────────────────────────────────
if ! python3 -m py_compile "$FILE_PATH" 2>&1; then
  echo "py_compile FAILED: $BASENAME — fix syntax errors before proceeding."
  exit 1
fi
echo "py_compile PASSED: $BASENAME"

# ── Check 2: AST structural check ────────────────────────────────────
# Verify: main() exists, argparse has --SANITY/--FULL, no bare imports
AST_RESULT=$(python3 -c "
import ast, sys

tree = ast.parse(open('$FILE_PATH').read())
funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
errors = []

# Every m*.py must have main()
if 'main' not in funcs:
    errors.append('missing main() function')

# Check for tqdm import (CLAUDE.md rule #13: mandatory for GPU scripts)
source = open('$FILE_PATH').read()
has_tqdm = 'from tqdm' in source or 'import tqdm' in source
# GPU modules need tqdm; CPU-only modules do not
# GPU: m04, m04d, m05, m05b, m05c, m06, m07, m09
# CPU: m00-m03, m04b, m04c, m06b, m08, m08b
cpu_only = {'m04b', 'm04c', 'm06b', 'm08', 'm08b'}
module_prefix = '$BASENAME'.split('_')[0]  # e.g. 'm04b', 'm09'
module_num = module_prefix.replace('m','').replace('b','').replace('c','').replace('d','')
is_gpu_module = (module_prefix not in cpu_only
                 and module_num.isdigit()
                 and int(module_num) >= 4)
if is_gpu_module and not has_tqdm:
    errors.append('missing tqdm import (CLAUDE.md rule #13: mandatory for GPU scripts)')

# Check argparse has --SANITY or --FULL
has_sanity = '--SANITY' in source
has_full = '--FULL' in source
if not has_sanity and not has_full:
    errors.append('missing --SANITY/--FULL argparse flags')

if errors:
    print('FAIL: ' + '; '.join(errors))
    sys.exit(1)
else:
    print('OK: main()=' + str(len(funcs)) + ' funcs, argparse=OK' + (', tqdm=OK' if has_tqdm else ''))
" 2>&1)

AST_EXIT=$?
if [ $AST_EXIT -ne 0 ]; then
  echo "AST check FAILED: $BASENAME — $AST_RESULT"
  exit 1
fi
echo "AST check PASSED: $BASENAME — $AST_RESULT"

# ── Check 3: ruff (undefined names, unused vars, redefined) ──────────
# Try direct ruff, then uvx ruff
RUFF_CMD=""
if command -v ruff &>/dev/null; then
    RUFF_CMD="ruff"
elif [ -x "$HOME/.local/bin/uvx" ]; then
    RUFF_CMD="$HOME/.local/bin/uvx ruff"
elif command -v uvx &>/dev/null; then
    RUFF_CMD="uvx ruff"
fi

if [ -n "$RUFF_CMD" ]; then
    RUFF_RESULT=$($RUFF_CMD check "$FILE_PATH" --select F821,F841,F811 2>&1)
    if echo "$RUFF_RESULT" | grep -q "^F8"; then
        echo "ruff FAILED: $BASENAME"
        echo "$RUFF_RESULT"
        exit 1
    fi
    echo "ruff PASSED: $BASENAME"
fi

exit 0
