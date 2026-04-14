#!/bin/bash
# Normal case (paste works) — unchanged:
# bash setup_claude.sh

# Paste-broken case (like today) — two commands:
# # On Mac:
# security find-generic-password -s "Claude Code-credentials" -w

# # On GPU (paste the JSON inside the quotes):
# export CLAUDE_CREDS_JSON='<paste JSON here>'
# bash setup_claude.sh

# The export command tolerates paste in web terminals (it's just shell input, not a TUI), so this path works
# everywhere.

# Security reminder: those tokens grant full Claude Max access. Before destroying the GPU instance, run rm
# ~/.claude/.credentials.json ~/.claude.json so the tokens don't linger in snapshots.


set -e  # stop on first error

# 1. Ensure curl is available
#   - macOS: curl is pre-installed, nothing to do
#   - Ubuntu/Debian (GPU devbox): install via apt
if ! command -v curl &> /dev/null; then
    if command -v apt &> /dev/null; then
        sudo apt update -y && sudo apt install -y curl
    else
        echo "ERROR: curl is not installed and no supported package manager found." >&2
        exit 1
    fi
fi

# 2. Install Claude Code via official native installer
#   (Node.js is NOT required — the native installer handles everything)
#   Docs: https://docs.anthropic.com/en/docs/claude-code/overview
curl -fsSL https://claude.ai/install.sh | bash

# Ensure ~/.local/bin is in PATH (native installer location)
export PATH="$HOME/.local/bin:$PATH"

# 3. (Optional) Paste-auth bypass for web terminals where the OAuth code
#    won't paste (Vast.ai web shell, some JupyterLab terminals, etc.).
#
#    How to use:
#      On your Mac, run:
#        security find-generic-password -s "Claude Code-credentials" -w
#      Copy the JSON it prints, then on the GPU box run:
#        export CLAUDE_CREDS_JSON='<paste JSON here>'
#        bash setup_claude.sh
#
#    If CLAUDE_CREDS_JSON is unset, this block is skipped and Claude Code
#    uses its normal interactive browser-based auth.
if [ -n "${CLAUDE_CREDS_JSON:-}" ]; then
    echo "Installing Claude credentials from CLAUDE_CREDS_JSON (paste-auth bypass)..."
    mkdir -p ~/.claude
    printf '%s\n' "$CLAUDE_CREDS_JSON" > ~/.claude/.credentials.json
    chmod 600 ~/.claude/.credentials.json
    # Skip the first-run "Select login method" welcome screen.
    [ -f ~/.claude.json ] || echo '{"hasCompletedOnboarding":true}' > ~/.claude.json
fi

# Navigate to your project directory.
# cd /path/to/your/project

# Launch Claude Code.
claude
