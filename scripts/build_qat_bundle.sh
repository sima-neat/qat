#!/usr/bin/env bash
set -euo pipefail

# Build-tool identity and package metadata must not be influenced by caller paths.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1

usage() {
  cat <<'USAGE'
Usage:
  build_qat_bundle.sh [--target-arch amd64|arm64|x86_64|aarch64] [--output-dir dist/ARCH] [--bundle-version VERSION]

Without --output-dir, the bundle is written to dist/<normalized-arch> under
the repository root. Relative paths are resolved from the repository root.
The build requires Python 3.12.3, pip 24.0, setuptools 84.0.0, and wheel 0.48.0.
USAGE
}

OUTPUT_DIR=""
TARGET_ARCH=""
BUNDLE_VERSION="sdk_version.neat+branch.git-short-hash"
BUNDLE_VERSION_EXPLICIT="0"
SOURCE_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--output-dir requires a value" >&2
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --target-arch)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--target-arch requires a value" >&2
        exit 1
      fi
      TARGET_ARCH="$2"
      shift 2
      ;;
    --bundle-version)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--bundle-version requires a value" >&2
        exit 1
      fi
      BUNDLE_VERSION="$2"
      BUNDLE_VERSION_EXPLICIT="1"
      shift 2
      ;;
    --source-json)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--source-json requires a value" >&2
        exit 1
      fi
      SOURCE_JSON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "$SOURCE_JSON" ]]; then
  SOURCE_JSON="$SCRIPT_DIR/source.json"
elif [[ "$SOURCE_JSON" != /* ]]; then
  SOURCE_JSON="$REPO_ROOT/$SOURCE_JSON"
fi
if [[ ! -f "$SOURCE_JSON" ]]; then
  echo "source json file not found: $SOURCE_JSON" >&2
  exit 1
fi

LEGAL_RESOURCE_NAMES=(
  LICENSE LICENSE.txt LICENSE.md
  NOTICE NOTICE.txt NOTICE.md
)
LEGAL_SOURCE_FILES=()
for legal_name in "${LEGAL_RESOURCE_NAMES[@]}"; do
  legal_path="$REPO_ROOT/$legal_name"
  [[ -e "$legal_path" ]] || continue
  if [[ -L "$legal_path" || ! -f "$legal_path" ]]; then
    echo "Legal resource is not a regular file: $legal_path" >&2
    exit 1
  fi
  if [[ ! -s "$legal_path" ]]; then
    echo "Ignoring empty legal resource in development bundle: $legal_path" >&2
    continue
  fi
  LEGAL_SOURCE_FILES+=("$legal_path")
done

validate_build_python() {
  local candidate="$1"

  "$candidate" - <<'PY'
from importlib import metadata
import sys

if sys.version_info[:3] != (3, 12, 3):
    raise SystemExit(
        f"QAT bundles require Python 3.12.3; found {sys.version.split()[0]}"
    )


required = {"pip": "24.0", "setuptools": "84.0.0", "wheel": "0.48.0"}
for distribution, expected_version in required.items():
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError as error:
        raise SystemExit(f"Missing build dependency: {distribution}") from error
    if version != expected_version:
        raise SystemExit(
            f"{distribution}=={expected_version} is required; found {version}"
        )
print(
    f"Build Python validated: Python {sys.version.split()[0]}, "
    f"pip {metadata.version('pip')}, setuptools {metadata.version('setuptools')}, "
    f"wheel {metadata.version('wheel')}"
)
PY
}

resolve_build_python() {
  local candidate=""
  local candidates=()

  if [[ -n "${QAT_BUILD_PYTHON:-}" ]]; then
    if [[ -x "$QAT_BUILD_PYTHON" ]] && \
       validate_build_python "$QAT_BUILD_PYTHON" >/dev/null 2>&1; then
      printf '%s\n' "$QAT_BUILD_PYTHON"
      return 0
    fi
    echo "QAT_BUILD_PYTHON does not satisfy the QAT build contract: $QAT_BUILD_PYTHON" >&2
    return 1
  fi

  candidate="$(command -v python3 2>/dev/null || true)"
  [[ -z "$candidate" ]] || candidates+=("$candidate")
  candidates+=(
    /sdk-extensions/model-compiler/bin/python
    /sdk-add-on/model-compiler/bin/python
  )
  if [[ -n "${HOME:-}" ]]; then
    candidates+=(
      "$HOME/sdk-extensions/model-compiler/bin/python"
      "$HOME/sdk-add-on/model-compiler/bin/python"
    )
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && validate_build_python "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! BUILD_PYTHON="$(resolve_build_python)"; then
  echo "No Python 3.12.3 with pip 24.0, setuptools 84.0.0, and wheel 0.48.0 was found." >&2
  echo "Activate Model Compiler or set QAT_BUILD_PYTHON explicitly." >&2
  exit 1
fi
validate_build_python "$BUILD_PYTHON"

normalize_target_arch() {
  local raw="$1"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    "")
      case "$(uname -m 2>/dev/null || echo unknown)" in
        x86_64|amd64) echo "amd64" ;;
        aarch64|arm64) echo "arm64" ;;
        *) echo "" ;;
      esac
      ;;
    x86_64|amd64) echo "amd64" ;;
    aarch64|arm64) echo "arm64" ;;
    *) echo "" ;;
  esac
}

TARGET_ARCH="$(normalize_target_arch "$TARGET_ARCH")"
if [[ -z "$TARGET_ARCH" ]]; then
  echo "Unable to resolve target architecture. Use --target-arch amd64 or arm64." >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$REPO_ROOT/dist/$TARGET_ARCH"
elif [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
fi
OUTPUT_DIR="$($BUILD_PYTHON -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).absolute())' "$OUTPUT_DIR")"
case "$OUTPUT_DIR" in
  /|"$REPO_ROOT")
    echo "Refusing unsafe bundle output directory: $OUTPUT_DIR" >&2
    exit 1
    ;;
esac
OUTPUT_PARENT="$(dirname -- "$OUTPUT_DIR")"
OUTPUT_NAME="$(basename -- "$OUTPUT_DIR")"

SDK_VERSION="$($BUILD_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("sdk_version", ""))' "$SOURCE_JSON")"
if [[ -z "$SDK_VERSION" ]]; then
  echo "source json has no sdk_version: $SOURCE_JSON" >&2
  exit 1
fi
PACKAGE_VERSION="$($BUILD_PYTHON -c 'import re,sys; value=open(sys.argv[1], encoding="utf-8").read().strip(); re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value) or sys.exit(f"Invalid package version: {value!r}"); print(value)' "$REPO_ROOT/VERSION.in")"
if [[ "${GITHUB_REF_TYPE:-}" == "tag" && -n "${GITHUB_REF_NAME:-}" ]]; then
  GIT_TAG="$GITHUB_REF_NAME"
else
  GIT_TAG="$(git -C "$REPO_ROOT" tag --points-at HEAD 2>/dev/null | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n1 || true)"
fi
if [[ -n "$GIT_TAG" ]]; then
  if [[ ! "$GIT_TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Release tag must match vMAJOR.MINOR.PATCH: $GIT_TAG" >&2
    exit 1
  fi
  TAG_VERSION="${GIT_TAG#v}"
  if [[ "$TAG_VERSION" != "$PACKAGE_VERSION" ]]; then
    echo "Release tag $GIT_TAG does not match VERSION.in ($PACKAGE_VERSION)." >&2
    exit 1
  fi
  if [[ "$BUNDLE_VERSION_EXPLICIT" == "0" ]]; then
    BUNDLE_VERSION="$TAG_VERSION"
  fi
fi
if [[ "$BUNDLE_VERSION" == *"sdk_version"* ]]; then
  BUNDLE_VERSION="${BUNDLE_VERSION//sdk_version/$SDK_VERSION}"
fi
if [[ "$BUNDLE_VERSION" == *"branch.git"* ]]; then
  GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  [[ -n "$GIT_BRANCH" && "$GIT_BRANCH" != "HEAD" ]] || GIT_BRANCH="unknown-branch"
  GIT_BRANCH_SAFE="$(printf '%s' "$GIT_BRANCH" | sed -E 's/[^A-Za-z0-9._-]+/-/g')"
  BUNDLE_VERSION="${BUNDLE_VERSION//branch.git/$GIT_BRANCH_SAFE}"
fi
if [[ "$BUNDLE_VERSION" == *"short-hash"* ]]; then
  GIT_SHORT_HASH="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
  [[ -n "$GIT_SHORT_HASH" ]] || GIT_SHORT_HASH="unknown"
  BUNDLE_VERSION="${BUNDLE_VERSION//short-hash/$GIT_SHORT_HASH}"
fi

validate_output_directory() {
  if [[ ! -e "$OUTPUT_DIR" ]]; then
    return 0
  fi
  if [[ -L "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    echo "Bundle output is not a regular directory: $OUTPUT_DIR" >&2
    return 1
  fi
  if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 \
    ! -type f -print -quit | grep -q .; then
    echo "Bundle output contains a directory or special entry: $OUTPUT_DIR" >&2
    return 1
  fi
  if find "$OUTPUT_DIR" -maxdepth 1 -type f \
    ! -name '*.whl' \
    ! -name 'manifest.txt' \
    ! -name 'metadata.json' \
    ! -name 'source.json' \
    ! -name 'install_qat_wheels.sh' \
    ! -name 'smoke_test_qat.py' \
    ! -name 'LICENSE' \
    ! -name 'LICENSE.txt' \
    ! -name 'LICENSE.md' \
    ! -name 'NOTICE' \
    ! -name 'NOTICE.txt' \
    ! -name 'NOTICE.md' \
    -print -quit | grep -q .; then
    echo "Bundle output contains an unrelated file: $OUTPUT_DIR" >&2
    return 1
  fi
}

validate_output_directory

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sima-qat-build.XXXXXX")"
PUBLISH_STAGE=""
PUBLISH_BACKUP=""
cleanup() {
  case "$BUILD_ROOT" in
    "${TMPDIR:-/tmp}"/sima-qat-build.*) rm -rf -- "$BUILD_ROOT" ;;
    *) echo "Refusing to remove unexpected build directory: $BUILD_ROOT" >&2 ;;
  esac
  if [[ -n "$PUBLISH_STAGE" ]]; then
    case "$PUBLISH_STAGE" in
      "$OUTPUT_PARENT"/."$OUTPUT_NAME".stage.*) rm -rf -- "$PUBLISH_STAGE" ;;
      *) echo "Refusing to remove unexpected bundle stage: $PUBLISH_STAGE" >&2 ;;
    esac
  fi
  if [[ -n "$PUBLISH_BACKUP" ]]; then
    case "$PUBLISH_BACKUP" in
      "$OUTPUT_PARENT"/."$OUTPUT_NAME".backup.*) rm -rf -- "$PUBLISH_BACKUP" ;;
      *) echo "Preserving unexpected bundle backup: $PUBLISH_BACKUP" >&2 ;;
    esac
  fi
}
trap cleanup EXIT

BUILD_DIST="$BUILD_ROOT/dist"
BUILD_SOURCE="$BUILD_ROOT/source"
BUNDLE_STAGE="$BUILD_ROOT/bundle"
mkdir -p "$BUILD_DIST" "$BUILD_SOURCE" "$BUNDLE_STAGE"
cp \
  "$REPO_ROOT/setup.py" \
  "$REPO_ROOT/pyproject.toml" \
  "$REPO_ROOT/README.md" \
  "$REPO_ROOT/VERSION.in" \
  "$REPO_ROOT/requirements.txt" \
  "$REPO_ROOT/requirements-test.txt" \
  "$REPO_ROOT/requirements-distill.txt" \
  "$BUILD_SOURCE/"
cp -R "$REPO_ROOT/sima_qat" "$BUILD_SOURCE/"
if [[ ${#LEGAL_SOURCE_FILES[@]} -gt 0 ]]; then
  cp "${LEGAL_SOURCE_FILES[@]}" "$BUILD_SOURCE/"
fi
"$BUILD_PYTHON" -m pip wheel \
  --disable-pip-version-check \
  --no-build-isolation \
  --no-deps \
  --wheel-dir "$BUILD_DIST" \
  "$BUILD_SOURCE"

wheels=("$BUILD_DIST"/*.whl)
if [[ ${#wheels[@]} -ne 1 || ! -f "${wheels[0]}" ]]; then
  echo "Expected exactly one built QAT wheel; found ${#wheels[@]}." >&2
  exit 1
fi
cp "${wheels[0]}" "$BUNDLE_STAGE/"
cp "$SCRIPT_DIR/install_qat_wheels.sh" "$BUNDLE_STAGE/"
cp "$SCRIPT_DIR/smoke_test_qat.py" "$BUNDLE_STAGE/"
cp "$SOURCE_JSON" "$BUNDLE_STAGE/source.json"
if [[ ${#LEGAL_SOURCE_FILES[@]} -gt 0 ]]; then
  cp "${LEGAL_SOURCE_FILES[@]}" "$BUNDLE_STAGE/"
fi

"$BUILD_PYTHON" "$SCRIPT_DIR/generate_metadata.py" \
  --artifacts-dir "$BUNDLE_STAGE" \
  --output "$BUNDLE_STAGE/metadata.json" \
  --version "$BUNDLE_VERSION" \
  --description "SiMa.ai NEAT QAT extension ($TARGET_ARCH)" \
  --package-version "$PACKAGE_VERSION" \
  --strict-inventory

validate_output_directory
mkdir -p "$OUTPUT_PARENT"
PUBLISH_STAGE="$(mktemp -d "$OUTPUT_PARENT/.${OUTPUT_NAME}.stage.XXXXXX")"
cp "$BUNDLE_STAGE"/* "$PUBLISH_STAGE/"
chmod 755 "$PUBLISH_STAGE"

if [[ -e "$OUTPUT_DIR" ]]; then
  PUBLISH_BACKUP="$(mktemp -d "$OUTPUT_PARENT/.${OUTPUT_NAME}.backup.XXXXXX")"
  rmdir -- "$PUBLISH_BACKUP"
  if ! mv -- "$OUTPUT_DIR" "$PUBLISH_BACKUP"; then
    echo "Could not preserve the previous bundle; output was not changed." >&2
    exit 1
  fi
  if mv -- "$PUBLISH_STAGE" "$OUTPUT_DIR"; then
    PUBLISH_STAGE=""
  else
    echo "Could not activate the new bundle; restoring the previous bundle." >&2
    if mv -- "$PUBLISH_BACKUP" "$OUTPUT_DIR"; then
      PUBLISH_BACKUP=""
    else
      echo "Automatic restore failed; previous bundle remains at: $PUBLISH_BACKUP" >&2
      PUBLISH_BACKUP=""
    fi
    exit 1
  fi
else
  mv -- "$PUBLISH_STAGE" "$OUTPUT_DIR"
  PUBLISH_STAGE=""
fi

if [[ -n "$PUBLISH_BACKUP" ]]; then
  rm -rf -- "$PUBLISH_BACKUP"
  PUBLISH_BACKUP=""
fi

echo "QAT $TARGET_ARCH bundle atomically written to: $OUTPUT_DIR"
