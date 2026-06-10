#!/usr/bin/env bash
# Generic setup script for the SiMa QAT package.
#
# Creates a Python venv, installs the runtime dependencies (from requirements.txt,
# the single source of truth for pinned versions), installs this package in
# editable mode, adds the test tooling, then runs a quick import / CUDA check.
#
# Usage:
#   ./setup_env.sh [VENV_DIR] [PYTHON_VERSION]
#
# Examples:
#   ./setup_env.sh                 # -> .venv,    python 3.12
#   ./setup_env.sh .venv311 3.11   # -> .venv311, python 3.11
set -euo pipefail

# Resolve the repo root from this script's location so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${1:-.venv}"
PYTHON_VERSION="${2:-3.12}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"

echo "[1/4] create venv ($VENV_DIR, python $PYTHON_VERSION)"
rm -rf "$VENV_DIR"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
PY="$VENV_DIR/bin/python"
$PY --version

echo "[2/4] install runtime dependencies (requirements.txt)"
# pyyaml/setuptools/wheel are build-time requirements for the editable install below
# (setup.py imports yaml and we use --no-build-isolation), so install them up front.
uv pip install --python "$PY" -r requirements.txt pyyaml setuptools wheel

echo "[3/4] install sima-qat (editable, no build isolation, no deps)"
uv pip install --python "$PY" --no-build-isolation --no-deps -e .

echo "[4/4] install test tooling"
uv pip install --python "$PY" pytest==7.0.0 pytest-cov pytest-timeout pytest-randomly pytest-xdist

echo "=== environment check ==="
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))"
echo "SETUP_DONE_OK"
