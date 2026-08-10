#!/usr/bin/env bash
# =============================================================================
# one_time_setup.sh
# Run this ONCE inside Ubuntu WSL as a user with sudo access.
# This installs the system packages that require sudo.
#
# HOW TO RUN:
#   1. Open Windows Terminal
#   2. Click the dropdown arrow → Ubuntu
#   3. Run: bash /mnt/c/Users/aksha/OneDrive/Documents/TEST_PRIORITY/scripts/one_time_setup.sh
# =============================================================================
set -euo pipefail

echo "Installing required system packages (requires sudo)..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    bash \
    dos2unix \
    python3.14-venv \
    python3-venv \
    git \
    curl \
    wget \
    build-essential

echo ""
echo "All done! You can now run the main setup script:"
echo "  bash /mnt/c/Users/aksha/OneDrive/Documents/TEST_PRIORITY/scripts/setup_wsl_no_sudo.sh"
