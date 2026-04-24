#!/bin/bash
# PreToolUse hook (GENERIC): block in-place edits to files currently being
# executed by a live process. Catches the "bash-inode hazard" class where
# sed -i / Edit / Write creates a new inode but the running interpreter keeps
# reading from the old inode held open via its fd. See:
# ~/.claude/projects/-workspace-factorjepa/memory/feedback_never_edit_running_scripts.md
#
# Detection: scan ps -eo pid,args for any live process whose cmdline contains
# the target file's basename. If found, BLOCK with a clear 3-resolution msg.
#
# Generic by design: no project-specific paths or filenames. Portable to any
# codebase where live scripts collide with in-place edits.
set -eo pipefail

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')

FILE=""
if [ "$TOOL" = "Edit" ] || [ "$TOOL" = "Write" ]; then
    FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
elif [ "$TOOL" = "Bash" ]; then
    CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
    if echo "$CMD" | grep -qE '\bsed\s+(-[a-z]*i[a-z]*|--in-place)\b'; then
        FILE=$(echo "$CMD" | grep -oE '[^[:space:]]+\.(sh|py|yaml|yml|json|toml|ini|cfg|conf)(\s|$)' | tail -1 | tr -d ' ')
    fi
fi

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
    exit 0
fi

BASENAME=$(basename "$FILE")
command -v ps >/dev/null 2>&1 || exit 0

ALL_PROCS=$(ps -eo pid,args --no-headers 2>/dev/null)
[ -z "$ALL_PROCS" ] && exit 0

# Substring match via bash built-ins (no pipe exit-code tricks).
MATCHES=""
EXCLUDE_PATTERN="claude-code|claude-cli|/node |grep |/tee |/cat |/less |/tail |/head |/wc |block-edit-running-file|post-edit-lint|enforce-dev-rules|fail-hard-research|protect-checkpoints"
while IFS= read -r line; do
    [ -z "$line" ] && continue
    [[ "$line" != *"$BASENAME"* ]] && continue
    [[ "$line" =~ $EXCLUDE_PATTERN ]] && continue
    MATCHES+="$line"$'\n'
done <<< "$ALL_PROCS"

if [ -z "$MATCHES" ]; then
    exit 0
fi

{
    echo "BLOCKED: '$FILE' is currently being executed by another process."
    echo ""
    echo "In-place edits on a running script are a TEXTBOOK INODE HAZARD:"
    echo "  - Edit/sed -i writes a new file and renames over the path (new inode)."
    echo "  - Running process holds the OLD inode via its open file descriptor."
    echo "  - Kernel keeps old inode alive until that fd closes."
    echo "  - Your edit DOES NOT TAKE EFFECT for the current run; only future runs see it."
    echo ""
    echo "Live process(es) using this file:"
    echo "$MATCHES"
    echo "Three safe resolutions:"
    echo "  1. KILL + RESTART: kill <pid>; edit; restart the process"
    echo "  2. NEW FILE:       cp to a v2 name; edit the copy; invoke after current run ends"
    echo "  3. WAIT:           let the running process finish naturally, THEN edit"
} >&2

exit 2
