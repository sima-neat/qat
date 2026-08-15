#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$SCRIPT_DIR"
SOURCE_JSON="$BUNDLE_DIR/source.json"
WHEEL_MANIFEST="$BUNDLE_DIR/manifest.txt"
SMOKE_TEST="$BUNDLE_DIR/smoke_test_qat.py"

find_model_compiler_dir() {
  local candidate=""

  if [[ -n "${QAT_MODEL_COMPILER_DIR:-}" ]]; then
    printf '%s\n' "$QAT_MODEL_COMPILER_DIR"
    return 0
  fi

  for candidate in \
    /sdk-extensions/model-compiler \
    /sdk-add-on/model-compiler \
    "$HOME/sdk-extensions/model-compiler"
  do
    if [[ -x "$candidate/bin/python" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

validate_model_compiler_environment() {
  local model_compiler_dir="$1"

  "$model_compiler_dir/bin/python" - "$SOURCE_JSON" "$model_compiler_dir" <<'PY'
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

source_json = Path(sys.argv[1])
expected_prefix = Path(sys.argv[2]).resolve()
with source_json.open(encoding="utf-8") as source_file:
    source = json.load(source_file)

errors = []
if Path(sys.prefix).resolve() != expected_prefix:
    errors.append(
        f"Python prefix is {Path(sys.prefix).resolve()}, expected {expected_prefix}"
    )

raw_python_version = str(source.get("python_version", ""))
try:
    expected_python = tuple(int(part) for part in raw_python_version.split("."))
except ValueError:
    expected_python = ()
if len(expected_python) != 3:
    errors.append(f"Invalid python_version in {source_json}: {raw_python_version!r}")
elif sys.version_info[:3] != expected_python:
    actual = ".".join(str(part) for part in sys.version_info[:3])
    errors.append(f"Python is {actual}, expected {raw_python_version}")

shared_environment = source.get("shared_environment", {})
expected_packages = shared_environment.get("packages", {})
if shared_environment.get("provider") != "model-compiler":
    errors.append(
        "source.json does not declare model-compiler as the environment provider"
    )
if not isinstance(expected_packages, dict) or not expected_packages:
    errors.append("source.json has no shared-environment package contract")
    expected_packages = {}

actual_versions = {}
for distribution, expected_version in expected_packages.items():
    try:
        actual_version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        errors.append(f"{distribution} is not installed (expected {expected_version})")
        continue
    actual_versions[distribution] = actual_version
    actual_base = actual_version.split("+", 1)[0]
    expected_base = str(expected_version).split("+", 1)[0]
    if actual_base != expected_base:
        errors.append(
            f"{distribution} is {actual_version}, expected base version {expected_base}"
        )

modules = {
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "pytorch-lightning": "pytorch_lightning",
}
for distribution in expected_packages:
    module = modules.get(distribution)
    if module is None:
        continue
    try:
        importlib.import_module(module)
    except Exception as error:
        errors.append(f"{distribution} cannot be imported: {error}")

if errors:
    print(
        "The Model Compiler environment is not compatible with this QAT artifact:",
        file=sys.stderr,
    )
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print(
        "QAT will not create another environment or change Python, Torch, or "
        "torchvision in the Model Compiler environment.",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(f"Compatible Model Compiler environment: {expected_prefix}")
print(f"Python: {sys.version.split()[0]}")
for distribution in expected_packages:
    print(f"{distribution}: {actual_versions[distribution]}")
PY
}

for required_file in "$SOURCE_JSON" "$WHEEL_MANIFEST" "$SMOKE_TEST"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing QAT bundle resource: $required_file" >&2
    exit 1
  fi
done

if ! MODEL_COMPILER_DIR="$(find_model_compiler_dir)"; then
  echo "Install Model Compiler before installing QAT." >&2
  echo "No model-compiler environment was found under /sdk-extensions or /sdk-add-on." >&2
  exit 1
fi

validate_model_compiler_environment "$MODEL_COMPILER_DIR"
if [[ "${QAT_VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "Model Compiler environment validation complete."
  exit 0
fi

wheels=()
while IFS= read -r entry; do
  [[ -n "$entry" ]] || continue
  if [[ ! -f "$BUNDLE_DIR/$entry" ]]; then
    echo "Wheel listed in manifest is missing: $entry" >&2
    exit 1
  fi
  wheels+=("$BUNDLE_DIR/$entry")
done < "$WHEEL_MANIFEST"

if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "No QAT wheel is listed in $WHEEL_MANIFEST." >&2
  exit 1
fi

echo "Validating the QAT wheel without resolving or changing dependencies..."
"$MODEL_COMPILER_DIR/bin/python" -m pip install \
  --disable-pip-version-check \
  --dry-run \
  --no-deps \
  "${wheels[@]}"

if "$MODEL_COMPILER_DIR/bin/python" -m pip show swml-qat >/dev/null 2>&1; then
  echo "Removing legacy swml-qat before installing sima-qat."
  "$MODEL_COMPILER_DIR/bin/python" -m pip uninstall -y swml-qat
fi

echo "Installing QAT into: $MODEL_COMPILER_DIR"
"$MODEL_COMPILER_DIR/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --force-reinstall \
  "${wheels[@]}"

"$MODEL_COMPILER_DIR/bin/python" -m pip check

echo "Running the installed-environment functional QAT smoke test..."
"$MODEL_COMPILER_DIR/bin/python" "$SMOKE_TEST" \
  --expected-prefix "$MODEL_COMPILER_DIR"

echo "QAT installation complete in the Model Compiler environment."
echo "Use activate-model-compiler to run QAT."
echo "Reinstall QAT after any Model Compiler reinstall or upgrade."
