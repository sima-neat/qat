#!/usr/bin/env bash
# Create the contributor-only PyTorch 2.8 CPU control environment.
#
# This is not the Model Compiler installation path. Customers install the QAT
# bundle into the existing Model Compiler environment with sima-cli.
#
# Usage: ./setup_env.sh [NEW_VENV_DIR]
# Default: build/venvs/torch28-control
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_VENV="${1:-$SCRIPT_DIR/build/venvs/torch28-control}"
CONTROL_PYTHON_VERSION="3.12.3"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"

if [[ -z "$CONTROL_VENV" || "$CONTROL_VENV" == "/" || "$CONTROL_VENV" == "$SCRIPT_DIR" ]]; then
  echo "Refusing unsafe control-environment path: $CONTROL_VENV" >&2
  exit 1
fi
if [[ -e "$CONTROL_VENV" ]]; then
  echo "Control-environment path already exists: $CONTROL_VENV" >&2
  echo "Choose a new path, or remove the existing environment explicitly." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create the PyTorch 2.8 control environment." >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/sima-qat-uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"

echo "[1/5] Create Python $CONTROL_PYTHON_VERSION environment: $CONTROL_VENV"
uv venv --python "$CONTROL_PYTHON_VERSION" "$CONTROL_VENV"
CONTROL_PYTHON="$CONTROL_VENV/bin/python"

echo "[2/5] Install the PyTorch 2.8 CPU control pair"
uv pip install --python "$CONTROL_PYTHON" \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"

echo "[3/5] Install pinned non-Torch runtime and test dependencies"
uv pip install --python "$CONTROL_PYTHON" \
  -r "$SCRIPT_DIR/requirements-torch28-control.txt" \
  -r "$SCRIPT_DIR/requirements-test.txt"

echo "[4/5] Install the checkout without resolving the Model Compiler pins"
uv pip install --python "$CONTROL_PYTHON" --no-deps -e "$SCRIPT_DIR"

echo "[5/5] Validate the control profile"
"$CONTROL_PYTHON" - <<'PY'
import importlib.metadata as metadata
import sys

import sima_qat
import torch

assert sys.version_info[:3] == (3, 12, 3), sys.version
assert metadata.version("torch").split("+", 1)[0] == "2.8.0"
assert metadata.version("torchvision").split("+", 1)[0] == "0.23.0"
print(
    "Torch 2.8 control ready:",
    f"python={sys.version.split()[0]}",
    f"torch={torch.__version__}",
    f"cuda={torch.cuda.is_available()}",
)
PY
uv pip check --python "$CONTROL_PYTHON"

echo "Run: $CONTROL_PYTHON -m pytest -q tests/integration"
