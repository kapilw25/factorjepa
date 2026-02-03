#!/bin/bash
set -e  # stop on first error

# 1. Update & install curl
sudo apt update -y && \
sudo apt install -y curl && \

# 2. Install Node.js (LTS) and npm
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && \
sudo apt install -y nodejs && \

# 3. Install Claude Code globally
sudo npm install -g @anthropic-ai/claude-code && \

# Navigate to your project directory.
# cd /path/to/your/project

# Launch Claude Code.
claude