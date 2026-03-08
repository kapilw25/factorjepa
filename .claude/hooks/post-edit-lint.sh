#!/bin/bash
# PostToolUse hook: Auto-run py_compile after Edit/Write on src/m*.py files.
# Fails (exit 1) if py_compile fails, forcing Claude to fix before proceeding.

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

# Run py_compile
if python3 -m py_compile "$FILE_PATH" 2>&1; then
  echo "py_compile PASSED: $BASENAME"
else
  echo "py_compile FAILED: $BASENAME — fix syntax errors before proceeding."
  exit 1
fi

exit 0
