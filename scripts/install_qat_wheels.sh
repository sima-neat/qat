#!/usr/bin/env bash
set -euo pipefail

# Never let the invoking checkout, user site, or caller PYTHONPATH influence
# environment identity, dependency checks, staging, or installed-wheel smoke.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$SCRIPT_DIR"
METADATA_JSON="$BUNDLE_DIR/metadata.json"
SOURCE_JSON="$BUNDLE_DIR/source.json"
WHEEL_MANIFEST="$BUNDLE_DIR/manifest.txt"
SMOKE_TEST="$BUNDLE_DIR/smoke_test_qat.py"

canonical_directory() {
  (cd "$1" 2>/dev/null && pwd -P)
}

is_known_model_compiler_location() {
  local candidate="$1"
  local known=""
  local locations=(
    /sdk-extensions/model-compiler
    /sdk-add-on/model-compiler
  )
  if [[ -n "${HOME:-}" ]]; then
    locations+=(
      "$HOME/sdk-extensions/model-compiler"
      "$HOME/sdk-add-on/model-compiler"
    )
  fi

  for known in "${locations[@]}"; do
    [[ -d "$known" ]] || continue
    if [[ "$(canonical_directory "$known")" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

find_model_compiler_dir() {
  local candidate=""
  local resolved=""
  local -a candidates=()
  local -A seen=()

  if [[ -n "${QAT_MODEL_COMPILER_DIR:-}" ]]; then
    if [[ ! -x "$QAT_MODEL_COMPILER_DIR/bin/python" ]]; then
      echo "QAT_MODEL_COMPILER_DIR has no executable bin/python: $QAT_MODEL_COMPILER_DIR" >&2
      return 1
    fi
    canonical_directory "$QAT_MODEL_COMPILER_DIR"
    return 0
  fi

  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    resolved="$(canonical_directory "$VIRTUAL_ENV")"
    if [[ "$(basename "$resolved")" == "model-compiler" ]] || \
       is_known_model_compiler_location "$resolved"; then
      printf '%s\n' "$resolved"
      return 0
    fi
  fi

  candidates=(
    /sdk-extensions/model-compiler
    /sdk-add-on/model-compiler
  )
  if [[ -n "${HOME:-}" ]]; then
    candidates+=(
      "$HOME/sdk-extensions/model-compiler"
      "$HOME/sdk-add-on/model-compiler"
    )
  fi

  resolved=""
  for candidate in "${candidates[@]}"; do
    [[ -x "$candidate/bin/python" ]] || continue
    candidate="$(canonical_directory "$candidate")"
    [[ -z "${seen[$candidate]:-}" ]] || continue
    seen[$candidate]=1
    if [[ -n "$resolved" ]]; then
      echo "Multiple Model Compiler environments were found:" >&2
      for candidate in "${!seen[@]}"; do
        echo "  - $candidate" >&2
      done
      echo "Set QAT_MODEL_COMPILER_DIR to the intended environment." >&2
      return 1
    fi
    resolved="$candidate"
  done

  [[ -n "$resolved" ]] || return 1
  printf '%s\n' "$resolved"
}

validate_bundle_checksums() {
  local python="$1"

  "$python" - "$METADATA_JSON" "$BUNDLE_DIR" <<'PY'
import hashlib
import hmac
import json
from pathlib import Path
import re
import sys

metadata_path = Path(sys.argv[1])
bundle_dir = Path(sys.argv[2]).resolve()
try:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"Cannot read bundle metadata {metadata_path}: {error}")

resources = metadata.get("resources")
checksums = metadata.get("resources-checksum")
if not isinstance(resources, list) or not resources:
    raise SystemExit("metadata.json has no non-empty resources list")
if len(resources) != len(set(resources)):
    raise SystemExit("metadata.json contains duplicate resources")
if not isinstance(checksums, dict) or set(checksums) != set(resources):
    raise SystemExit("metadata resource/checksum inventory does not match")

for name in resources:
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise SystemExit(f"Unsafe bundle resource name: {name!r}")
    path = bundle_dir / name
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"Bundle resource is missing or is not a regular file: {name}")
    expected = checksums[name]
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise SystemExit(f"Invalid SHA-256 value for bundle resource: {name}")
    digest = hashlib.sha256()
    with path.open("rb") as resource_file:
        for chunk in iter(lambda: resource_file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise SystemExit(
            f"Checksum mismatch for {name}: expected {expected}, found {actual}"
        )

actual_inventory = {path.name for path in bundle_dir.iterdir()}
expected_inventory = set(resources) | {metadata_path.name}
if actual_inventory != expected_inventory:
    raise SystemExit(
        "Bundle inventory mismatch: "
        f"missing={sorted(expected_inventory - actual_inventory)}, "
        f"unexpected={sorted(actual_inventory - expected_inventory)}"
    )

print(f"Verified SHA-256 checksums for {len(resources)} bundle resources.")
PY
}

validate_model_compiler_environment() {
  local model_compiler_dir="$1"

  "$model_compiler_dir/bin/python" - "$SOURCE_JSON" "$model_compiler_dir" <<'PY'
import importlib
import importlib.metadata
import importlib.util
import json
import os
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
identity_problems = []
try:
    afe_spec = importlib.util.find_spec("afe")
except (ImportError, AttributeError, ValueError) as error:
    afe_spec = None
    identity_problems.append(f"cannot inspect afe: {error}")
if afe_spec is None:
    identity_problems.append("afe module is missing")
try:
    sima_mlc_version = importlib.metadata.version("sima-mlc")
except importlib.metadata.PackageNotFoundError:
    sima_mlc_version = None
    identity_problems.append("sima-mlc distribution is missing")
is_model_compiler = not identity_problems
allow_test_fixture = (
    os.environ.get("QAT_ALLOW_COMPATIBLE_TEST_ENV") == "1"
    and os.environ.get("CI") == "true"
)
if identity_problems:
    identity_message = "; ".join(identity_problems)
    if allow_test_fixture:
        print(
            "WARNING: accepting a test-only compatible CI environment "
            f"({identity_message}); this override is not a supported customer path.",
            file=sys.stderr,
        )
    else:
        errors.append(
            f"this is not a Model Compiler environment: {identity_message}"
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

environment_label = (
    "Model Compiler" if is_model_compiler else "test-only compatible CI fixture"
)
print(f"Compatible {environment_label}: {expected_prefix}")
print(f"Python: {sys.version.split()[0]}")
if sima_mlc_version is not None:
    print(f"sima-mlc: {sima_mlc_version}")
for distribution in expected_packages:
    print(f"{distribution}: {actual_versions[distribution]}")
PY
}

backup_distribution() {
  local distribution="$1"
  local archive="$2"

  "$MODEL_COMPILER_PYTHON" - "$distribution" "$archive" "$MODEL_COMPILER_DIR" <<'PY'
import importlib.metadata
from pathlib import Path
import sys
import tarfile

distribution_name = sys.argv[1]
archive = Path(sys.argv[2])
prefix = Path(sys.argv[3]).resolve()
distribution = importlib.metadata.distribution(distribution_name)
files = distribution.files
if files is None:
    raise SystemExit(f"Cannot enumerate installed files for {distribution_name}")

backed_up = 0
with tarfile.open(archive, "w:gz") as output:
    for entry in files:
        source = Path(distribution.locate_file(entry))
        if not source.exists():
            continue
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(prefix)
        except ValueError as error:
            raise SystemExit(
                f"Refusing to back up {distribution_name} file outside {prefix}: {resolved}"
            ) from error
        if source.is_symlink():
            raise SystemExit(
                f"Refusing unsupported symlink in {distribution_name}: {source}"
            )
        output.add(resolved, arcname=relative, recursive=False)
        backed_up += 1
if backed_up == 0:
    raise SystemExit(f"No installed files were found for {distribution_name}")
print(f"Backed up {distribution_name} ({backed_up} files).")
PY
}

restore_distribution() {
  local archive="$1"
  [[ -f "$archive" ]] || return 0

  "$MODEL_COMPILER_PYTHON" - "$archive" "$MODEL_COMPILER_DIR" <<'PY'
from pathlib import Path
import sys
import tarfile

archive = Path(sys.argv[1])
prefix = Path(sys.argv[2]).resolve()
with tarfile.open(archive, "r:gz") as backup:
    backup.extractall(prefix, filter="data")
PY
}

rollback_installation() {
  local original_status="$1"
  local rollback_failed=0

  set +e
  echo "QAT installation failed after package mutation; attempting rollback." >&2
  "$MODEL_COMPILER_PYTHON" -m pip uninstall -y sima-qat >/dev/null 2>&1
  if [[ -f "$PREVIOUS_SIMA_BACKUP" ]]; then
    restore_distribution "$PREVIOUS_SIMA_BACKUP" || rollback_failed=1
  fi
  if [[ -f "$LEGACY_BACKUP" ]]; then
    restore_distribution "$LEGACY_BACKUP" || rollback_failed=1
  fi
  "$MODEL_COMPILER_PYTHON" -m pip check >/dev/null 2>&1 || rollback_failed=1
  if [[ "$rollback_failed" == "0" ]]; then
    echo "The previous QAT package state was restored." >&2
  else
    echo "Automatic rollback was incomplete." >&2
    echo "Repair or reinstall Model Compiler, then rerun this installer." >&2
  fi
  set -e
  return "$original_status"
}

if [[ ! -f "$METADATA_JSON" || -L "$METADATA_JSON" ]]; then
  echo "Missing or unsafe QAT bundle metadata: $METADATA_JSON" >&2
  exit 1
fi

if ! MODEL_COMPILER_DIR="$(find_model_compiler_dir)"; then
  echo "Install or activate Model Compiler before installing QAT." >&2
  echo "No unambiguous model-compiler environment was found." >&2
  echo "Set QAT_MODEL_COMPILER_DIR if more than one installation exists." >&2
  exit 1
fi
MODEL_COMPILER_PYTHON="$MODEL_COMPILER_DIR/bin/python"

validate_bundle_checksums "$MODEL_COMPILER_PYTHON"
validate_model_compiler_environment "$MODEL_COMPILER_DIR"
"$MODEL_COMPILER_PYTHON" -m pip check

wheels=()
while IFS= read -r entry || [[ -n "$entry" ]]; do
  [[ -n "$entry" ]] || continue
  if [[ "$(basename -- "$entry")" != "$entry" || "$entry" != *.whl ]]; then
    echo "Unsafe wheel manifest entry: $entry" >&2
    exit 1
  fi
  if [[ ! -f "$BUNDLE_DIR/$entry" || -L "$BUNDLE_DIR/$entry" ]]; then
    echo "Wheel listed in manifest is missing or unsafe: $entry" >&2
    exit 1
  fi
  wheels+=("$BUNDLE_DIR/$entry")
done < "$WHEEL_MANIFEST"

if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "Expected exactly one QAT wheel in $WHEEL_MANIFEST; found ${#wheels[@]}." >&2
  exit 1
fi
INSTALL_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sima-qat-install.XXXXXX")"
cleanup() {
  case "$INSTALL_ROOT" in
    "${TMPDIR:-/tmp}"/sima-qat-install.*) rm -rf -- "$INSTALL_ROOT" ;;
    *) echo "Refusing to remove unexpected install staging directory: $INSTALL_ROOT" >&2 ;;
  esac
}
trap cleanup EXIT
STAGED_SITE="$INSTALL_ROOT/site"
PREVIOUS_SIMA_BACKUP="$INSTALL_ROOT/previous-sima-qat.tar.gz"
LEGACY_BACKUP="$INSTALL_ROOT/legacy-swml-qat.tar.gz"
mkdir -p "$STAGED_SITE"

printf '%s\n' "Checking wheel dependency metadata against Model Compiler..."
"$MODEL_COMPILER_PYTHON" -m pip install \
  --disable-pip-version-check \
  --dry-run \
  --no-index \
  "${wheels[@]}"

printf '%s\n' "Staging and functionally validating the wheel before package mutation..."
"$MODEL_COMPILER_PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-compile \
  --no-deps \
  --no-index \
  --target "$STAGED_SITE" \
  "${wheels[@]}"
(
  cd "$INSTALL_ROOT"
  env -u PYTHONPATH PYTHONPATH="$STAGED_SITE" PYTHONNOUSERSITE=1 \
    "$MODEL_COMPILER_PYTHON" -P "$SMOKE_TEST" \
    --expected-prefix "$MODEL_COMPILER_DIR"
)

if [[ "${QAT_VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "Bundle and Model Compiler validation complete; no packages were changed."
  exit 0
fi

had_sima=0
had_legacy=0
if "$MODEL_COMPILER_PYTHON" -m pip show sima-qat >/dev/null 2>&1; then
  had_sima=1
fi
if "$MODEL_COMPILER_PYTHON" -m pip show swml-qat >/dev/null 2>&1; then
  had_legacy=1
fi
if [[ "$had_sima" == "1" && "$had_legacy" == "1" ]]; then
  echo "Both sima-qat and legacy swml-qat are installed; package ownership is ambiguous." >&2
  echo "Repair the Model Compiler environment before installing this bundle." >&2
  exit 1
fi
if [[ "$had_sima" == "1" ]]; then
  backup_distribution sima-qat "$PREVIOUS_SIMA_BACKUP"
fi
if [[ "$had_legacy" == "1" ]]; then
  backup_distribution swml-qat "$LEGACY_BACKUP"
fi

if [[ "$had_legacy" == "1" ]]; then
  echo "Removing legacy swml-qat after successful staged validation."
  if "$MODEL_COMPILER_PYTHON" -m pip uninstall -y swml-qat; then
    :
  else
    status=$?
    rollback_installation "$status" || true
    exit "$status"
  fi
fi

echo "Installing QAT into: $MODEL_COMPILER_DIR"
if "$MODEL_COMPILER_PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --no-index \
  --force-reinstall \
  "${wheels[@]}"; then
  :
else
  status=$?
  rollback_installation "$status" || true
  exit "$status"
fi

if "$MODEL_COMPILER_PYTHON" -m pip check; then
  :
else
  status=$?
  rollback_installation "$status" || true
  exit "$status"
fi

printf '%s\n' "Running the installed-environment functional QAT smoke test..."
if (
  cd "$INSTALL_ROOT" &&
    env -u PYTHONPATH PYTHONNOUSERSITE=1 \
      "$MODEL_COMPILER_PYTHON" -P "$SMOKE_TEST" \
      --expected-prefix "$MODEL_COMPILER_DIR"
); then
  :
else
  status=$?
  rollback_installation "$status" || true
  exit "$status"
fi

echo "QAT installation complete in the Model Compiler environment."
echo "Use activate-model-compiler to run QAT."
echo "Reinstall QAT after any Model Compiler reinstall or upgrade."
