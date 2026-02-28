#!/bin/bash
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

# Navigate to your project directory.
# cd /path/to/your/project

# Launch Claude Code.
claude
