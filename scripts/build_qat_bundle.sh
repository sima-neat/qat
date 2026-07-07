#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_qat_bundle.sh [--output-dir ./dist/amd64] [--target-arch amd64|arm64|x86_64|aarch64] [--bundle-version VERSION]
EOF
}

OUTPUT_DIR="./dist/qat"
TARGET_ARCH=""
BUNDLE_VERSION="sdk_version.neat+branch.git-short-hash"
BUNDLE_VERSION_EXPLICIT="0"
SOURCE_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --target-arch) TARGET_ARCH="${2:-}"; shift 2 ;;
    --bundle-version) BUNDLE_VERSION="${2:-}"; BUNDLE_VERSION_EXPLICIT="1"; shift 2 ;;
    --source-json) SOURCE_JSON="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "$SOURCE_JSON" ]]; then
  SOURCE_JSON="$SCRIPT_DIR/source.json"
fi
if [[ ! -f "$SOURCE_JSON" ]]; then
  echo "source json file not found: $SOURCE_JSON" >&2
  exit 1
fi

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
  echo "Unable to resolve target architecture. Use --target-arch amd64 or --target-arch arm64." >&2
  exit 1
fi

SDK_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("sdk_version", ""))' "$SOURCE_JSON")"
if [[ "$BUNDLE_VERSION_EXPLICIT" == "0" ]]; then
  GIT_TAG="$(git -C "$REPO_ROOT" tag --points-at HEAD 2>/dev/null | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n1 || true)"
  if [[ -n "$GIT_TAG" ]]; then
    BUNDLE_VERSION="${GIT_TAG#v}"
  fi
fi
if [[ "$BUNDLE_VERSION" == *"sdk_version"* ]]; then
  BUNDLE_VERSION="${BUNDLE_VERSION//sdk_version/$SDK_VERSION}"
fi
if [[ "$BUNDLE_VERSION" == *"branch.git"* ]]; then
  GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  [[ -n "$GIT_BRANCH" && "$GIT_BRANCH" != "HEAD" ]] || GIT_BRANCH="unknown-branch"
  GIT_BRANCH_SAFE="$(echo "$GIT_BRANCH" | sed -E 's/[^A-Za-z0-9._-]+/-/g')"
  BUNDLE_VERSION="${BUNDLE_VERSION//branch.git/$GIT_BRANCH_SAFE}"
fi
if [[ "$BUNDLE_VERSION" == *"short-hash"* ]]; then
  GIT_SHORT_HASH="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
  [[ -n "$GIT_SHORT_HASH" ]] || GIT_SHORT_HASH="unknown"
  BUNDLE_VERSION="${BUNDLE_VERSION//short-hash/$GIT_SHORT_HASH}"
fi

rm -rf "$REPO_ROOT/build" "$REPO_ROOT/dist"
(
  cd "$REPO_ROOT"
  python3 setup.py bdist_wheel
)

mkdir -p "$OUTPUT_DIR"
echo "Cleaning generated QAT bundle artifacts from: $OUTPUT_DIR"
find "$OUTPUT_DIR" -maxdepth 1 -type f \( -name '*.whl' -o -name 'manifest.txt' -o -name 'metadata.json' -o -name 'source.json' -o -name 'install_qat_wheels.sh' \) -delete

cp "$REPO_ROOT"/dist/*.whl "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/install_qat_wheels.sh" "$OUTPUT_DIR/"
cp "$SOURCE_JSON" "$OUTPUT_DIR/source.json"

python3 "$SCRIPT_DIR/generate_metadata.py" \
  --artifacts-dir "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/metadata.json" \
  --version "$BUNDLE_VERSION" \
  --description "SiMa.ai NEAT QAT extension ($TARGET_ARCH)"

echo "QAT $TARGET_ARCH bundle written to: $OUTPUT_DIR"
