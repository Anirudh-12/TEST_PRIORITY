#!/usr/bin/env bash
# =============================================================================
# install_deadsnakes.sh
# Run this ONCE inside Ubuntu WSL as a user with sudo access.
# This adds the deadsnakes PPA and installs older Python versions needed for BugsInPy.
#
# HOW TO RUN:
#   1. Open Windows Terminal -> Ubuntu
#   2. Run: bash /mnt/c/Users/aksha/OneDrive/Documents/TEST_PRIORITY/scripts/install_deadsnakes.sh
# =============================================================================
set -euo pipefail

echo "Adding deadsnakes PPA..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -qq

echo "Installing Python 3.7 and 3.8 and their venv modules..."
sudo apt-get install -y --no-install-recommends \
    python3.7 python3.7-venv python3.7-dev \
    python3.8 python3.8-venv python3.8-dev

echo "Symlinking Python versions so bugsinpy-compile can use them..."
mkdir -p $HOME/.local/bin

# BugsInPy uses specific python names in its env creation sometimes,
# or relies on the active `python3`. We will configure checkout.py to
# put the right binary in the PATH.

ln -sf /usr/bin/python3.7 $HOME/.local/bin/python3.7
ln -sf /usr/bin/python3.8 $HOME/.local/bin/python3.8

echo "Python versions installed successfully from deadsnakes PPA!"
