#!/bin/bash
# PreToolUse hook: Block deletion of GPU training outputs without safeguards.
# Prevents accidental loss of expensive checkpoints (student_encoder.pt, *.npy embeddings).
#
# Triggered by: rm -f/rf commands targeting training outputs in Edit/Write/Bash tools.
# Rule: NEVER delete student_encoder.pt, embeddings*.npy, or training_summary.json
# without first checking if the file represents a completed multi-epoch run.
#
# Background: A 5-epoch student_encoder.pt (3h GPU, ~$2.40) was deleted by
# run_pretrain.sh's winner selector without checking epoch count. This hook
# prevents that class of mistake from recurring.

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
NEW_CONTENT=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')

# ── Check Bash commands for dangerous rm on training outputs ─────────
if [ "$TOOL" = "Bash" ] && [ -n "$COMMAND" ]; then
    # Block: rm -rf on directories containing training outputs
    if echo "$COMMAND" | grep -qE 'rm\s+(-rf?|-f)\s+.*m09_lambda'; then
        # Allow if the command is clearly part of a "clean corrupt outputs" flow
        # (e.g., after a confirmed failed run with disk full)
        echo "BLOCKED: Deleting m09_lambda* directory which may contain student_encoder.pt"
        echo "If this is intentional cleanup, explain why in your message first."
        echo "Cost: student_encoder.pt = 3.8GB model trained over hours of GPU time."
        exit 2
    fi

    # Block: rm on specific high-value files
    if echo "$COMMAND" | grep -qE 'rm\s+.*student_encoder\.pt'; then
        echo "BLOCKED: Deleting student_encoder.pt — this is a trained model checkpoint."
        echo "Check training_summary.json for epoch count before deleting."
        echo "If this encoder was trained for multiple epochs, deletion wastes hours of GPU."
        exit 2
    fi
fi

# ── Check shell scripts being written/edited for unguarded deletion ──
if [ "$TOOL" = "Edit" ] || [ "$TOOL" = "Write" ]; then
    FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

    if echo "$FILE_PATH" | grep -qE '\.sh$'; then
        # Block: rm -f student_encoder.pt without epoch check
        if echo "$NEW_CONTENT" | grep -qE 'rm\s+(-f\s+)?.*student_encoder'; then
            if ! echo "$NEW_CONTENT" | grep -qE 'training_summary|epoch|max_epochs'; then
                echo "BLOCKED: Shell script deletes student_encoder.pt without checking epoch count."
                echo "Add a guard: read training_summary.json, check epochs before deleting."
                echo "Background: 5-epoch checkpoint (3h GPU) was deleted by unguarded rm."
                exit 2
            fi
        fi
    fi
fi

exit 0
