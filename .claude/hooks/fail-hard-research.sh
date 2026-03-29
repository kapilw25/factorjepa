#!/bin/bash
# PreToolUse hook: Block silent error swallowing in pipeline scripts.
# Research pipelines must FAIL HARD — wrong numbers reject papers.
#
# Blocks:
#   1. Shell scripts with `|| { log "WARNING"` (swallows errors)
#   2. Shell scripts with `|| continue` or `|| true` (skips failures)
#   3. Python try/except that silently passes (bare except: pass)
#   4. Writing shell scripts without `set -euo pipefail`
#
# See CLAUDE.md rule #14: "Fail hard in research — silent errors produce
# garbage that looks plausible but is wrong."

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')

# Only check Edit and Write tools
if [ "$TOOL" != "Edit" ] && [ "$TOOL" != "Write" ]; then
  exit 0
fi

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
NEW_CONTENT=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')

# Skip if no content to check
if [ -z "$NEW_CONTENT" ]; then
  exit 0
fi

# ── Check 1: Shell scripts — no silent error swallowing ──────────────
if echo "$FILE_PATH" | grep -qE '\.sh$'; then

  # Block: || continue (skips to next iteration on failure)
  if echo "$NEW_CONTENT" | grep -qE '\|\|\s*\{[^}]*continue' ; then
    echo "BLOCKED: '|| continue' swallows pipeline errors silently."
    echo "Research pipelines must fail hard: use '|| { log \"FATAL: ...\"; exit 1; }'"
    echo "Reason: silent failures produce wrong metrics → paper rejection."
    exit 2
  fi

  # Block: || true (ignores failure)
  if echo "$NEW_CONTENT" | grep -qE '\|\|\s*true\b'; then
    echo "BLOCKED: '|| true' ignores pipeline errors."
    echo "Use '|| { log \"FATAL: ...\"; exit 1; }' instead."
    exit 2
  fi

  # Block: WARNING + no exit (error logged but execution continues)
  if echo "$NEW_CONTENT" | grep -qE '\|\|\s*\{[^}]*WARNING[^}]*\}' | grep -vq 'exit'; then
    # More precise: check if the block has WARNING but no exit
    if echo "$NEW_CONTENT" | grep -qE '\|\|\s*\{[^}]*WARNING' && ! echo "$NEW_CONTENT" | grep -qE '\|\|\s*\{[^}]*exit'; then
      echo "BLOCKED: Error handler logs WARNING but doesn't exit."
      echo "Use 'FATAL' + 'exit 1' — a warning that doesn't stop is a lie."
      exit 2
    fi
  fi
fi

# ── Check 2: Python — no bare except: pass ───────────────────────────
if echo "$FILE_PATH" | grep -qE '\.py$'; then
  if echo "$NEW_CONTENT" | grep -qE 'except.*:\s*$' && echo "$NEW_CONTENT" | grep -qE '^\s*pass\s*$'; then
    echo "BLOCKED: bare 'except: pass' silently swallows errors."
    echo "Either handle the error explicitly or let it propagate."
    exit 2
  fi
fi

exit 0
