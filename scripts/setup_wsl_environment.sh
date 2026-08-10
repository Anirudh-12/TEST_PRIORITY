#!/usr/bin/env bash
# =============================================================================
# setup_wsl_environment.sh
# Sets up the WSL Ubuntu environment for BugsInPy processing.
#
# Run this INSIDE Ubuntu WSL:
#   bash /mnt/c/Users/aksha/OneDrive/Documents/TEST_PRIORITY/scripts/setup_wsl_environment.sh
# =============================================================================
set -euo pipefail

WORKSPACE="/home/akshay/bugsinpy_workspace"
PROJECT_ROOT="/mnt/c/Users/aksha/OneDrive/Documents/TEST_PRIORITY"

echo "============================================================"
echo "  BugsInPy Research Environment Setup"
echo "============================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    libpq-dev \
    sqlite3 \
    jq \
    2>/dev/null
echo "      System packages OK"

# ── 2. Install uv ─────────────────────────────────────────────────────────────
echo "[2/6] Installing uv..."
if command -v uv &>/dev/null; then
    echo "      uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to current session PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    echo "      uv installed: $(uv --version)"
fi

# Persist uv in PATH
if ! grep -q "uv" ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

# ── 3. Install pyenv for Python version management ───────────────────────────
echo "[3/6] Installing pyenv..."
if command -v pyenv &>/dev/null; then
    echo "      pyenv already installed: $(pyenv --version)"
else
    curl -fsSL https://pyenv.run | bash
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    
    # Persist in .bashrc
    if ! grep -q "pyenv" ~/.bashrc 2>/dev/null; then
        cat >> ~/.bashrc << 'EOF'
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
    fi
    echo "      pyenv installed"
fi

# ── 4. Install Python versions used by BugsInPy ──────────────────────────────
echo "[4/6] Installing Python versions used by BugsInPy..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

# Most BugsInPy bugs use 3.6, 3.7, 3.8. Install those that work on Ubuntu 26.
for PY_VERSION in "3.8.20" "3.9.21" "3.10.16" "3.11.12"; do
    if pyenv versions --bare 2>/dev/null | grep -q "^${PY_VERSION}$"; then
        echo "      Python ${PY_VERSION} already installed"
    else
        echo "      Installing Python ${PY_VERSION}..."
        pyenv install "${PY_VERSION}" 2>&1 | tail -3 || echo "      WARNING: Python ${PY_VERSION} install failed"
    fi
done

# ── 5. Create workspace and clone BugsInPy ───────────────────────────────────
echo "[5/6] Setting up workspace..."
mkdir -p "${WORKSPACE}"

if [ -d "${WORKSPACE}/BugsInPy/.git" ]; then
    echo "      BugsInPy already cloned"
    cd "${WORKSPACE}/BugsInPy" && git pull --quiet || true
else
    echo "      Cloning BugsInPy..."
    git clone --depth=1 https://github.com/soarsmu/BugsInPy.git "${WORKSPACE}/BugsInPy"
fi

# Add BugsInPy to PATH
BUGSINPY_BIN="${WORKSPACE}/BugsInPy/framework/bin"
if ! grep -q "bugsinpy" ~/.bashrc 2>/dev/null; then
    echo "export PATH=\"\$PATH:${BUGSINPY_BIN}\"" >> ~/.bashrc
fi
export PATH="$PATH:${BUGSINPY_BIN}"

echo "      Workspace: ${WORKSPACE}"
echo "      BugsInPy:  ${WORKSPACE}/BugsInPy"

# ── 6. Install project Python dependencies ────────────────────────────────────
echo "[6/6] Installing project Python dependencies..."
if command -v uv &>/dev/null; then
    cd "${PROJECT_ROOT}" && uv sync 2>&1 | tail -5 || \
        pip install -r "${PROJECT_ROOT}/requirements.txt" --quiet
else
    pip install jsonschema scipy numpy pandas scikit-learn rich click pyyaml tqdm --quiet
fi

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Workspace: ${WORKSPACE}"
echo "  BugsInPy:  ${WORKSPACE}/BugsInPy"
echo ""
echo "  Test with:"
echo "    python -m bugsinpy.process --project thefuck --bug 1 --skip-coverage"
echo "============================================================"
