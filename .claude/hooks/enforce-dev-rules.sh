#!/bin/bash
# Enforce project development rules via PreToolUse hook.
# Blocks: pip install, git state changes, bare python3.
# See src/CLAUDE.md for the rules these enforce.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Skip if no command (non-Bash tool)
if [ -z "$COMMAND" ]; then
  exit 0
fi

# Rule 1: Block pip install / uv pip install
# Enforcement of: "NEVER install packages directly via pip install or uv pip install"
if echo "$COMMAND" | grep -qiE '(^|\s)(pip|uv\s+pip)\s+install'; then
  echo "BLOCKED: Direct pip/uv install not allowed. Add deps to requirements.txt then run: bash setup_env_uv.sh"
  exit 2
fi

# Rule 2: Block git commit, push, add, reset
# Enforcement of: "Git: provide commit message text only. NEVER run git commands."
if echo "$COMMAND" | grep -qiE '(^|\s)git\s+(commit|push|add|reset|rebase|cherry-pick|merge|tag)(\s|$)'; then
  echo "BLOCKED: Git state-change commands not allowed. Provide commit message text only — user runs git manually via git_push.sh"
  exit 2
fi

# Rule 3: Block bare python3 without venv activation
# Enforcement of: "ALWAYS activate the venv before running any python3 command"
# Allow: source venv_walkindia/bin/activate && python3 ...
# Allow: venv_walkindia/bin/python3 ...
# Block: python3 -m ... (no venv prefix)
if echo "$COMMAND" | grep -qE '(^|\s)python3\s+' ; then
  # Allow if command contains venv activation or full venv path
  if echo "$COMMAND" | grep -qE '(source\s+.*venv_|venv_walkindia/bin/)'; then
    exit 0
  fi
  # Allow py_compile and ast checks (M1 lint-only, no GPU needed)
  if echo "$COMMAND" | grep -qE 'python3\s+-(m\s+py_compile|m\s+ast|c\s+)'; then
    exit 0
  fi
  echo "BLOCKED: Bare python3 without venv activation. Use: source venv_walkindia/bin/activate && python3 ..."
  exit 2
fi

exit 0
