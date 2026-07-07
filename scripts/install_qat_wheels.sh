#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$SCRIPT_DIR"
SOURCE_JSON="$BUNDLE_DIR/source.json"
WHEEL_MANIFEST="$BUNDLE_DIR/manifest.txt"
EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-https://pypi.org/simple}"

read_source_json_field() {
  local expr="$1"
  python3 -c '
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    doc = json.load(f)
expr = sys.argv[2]
if expr == "python_version":
    value = doc.get("python_version", "")
    print(value if isinstance(value, str) else "")
' "$SOURCE_JSON" "$expr"
}

normalize_python_version() {
  local raw="$1"
  raw="$(echo "$raw" | tr -d "[:space:]")"
  if [[ "$raw" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "$raw"
    return 0
  fi
  if [[ "$raw" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "$(echo "$raw" | awk -F. '{print $1"."$2}')"
    return 0
  fi
  return 1
}

resolve_python_cmd() {
  local py_mm="$1"
  local major="${py_mm%%.*}"
  local minor="${py_mm##*.}"
  local cmd=""

  for candidate in "python${major}.${minor}" "python${major}" python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      cmd="$candidate"
      if "$cmd" -c "import sys; raise SystemExit(0 if (sys.version_info.major, sys.version_info.minor)==(${major},${minor}) else 1)" >/dev/null 2>&1; then
        echo "$cmd"
        return 0
      fi
    fi
  done

  if command -v pyenv >/dev/null 2>&1; then
    local target_version="$py_mm"
    local latest_patch=""
    latest_patch="$(pyenv install --list 2>/dev/null | sed 's/^[[:space:]]*//' | grep -E "^${py_mm//./\.}\.[0-9]+$" | sort -V | tail -n1 || true)"
    if [[ -n "$latest_patch" ]]; then
      target_version="$latest_patch"
    fi
    pyenv install -s "$target_version"
    echo "${PYENV_ROOT:-$HOME/.pyenv}/versions/$target_version/bin/python"
    return 0
  fi

  return 1
}

reset_venv_dir() {
  local venv_dir="$1"
  case "$venv_dir" in
    /sdk-extensions/qat|/sdk-add-on/qat|"$HOME"/sdk-extensions/qat) ;;
    *)
      echo "Refusing to reset unexpected QAT venv path: $venv_dir" >&2
      return 1
      ;;
  esac
  rm -rf "$venv_dir"
}

configure_shell_helpers() {
  local qat_dir="$1"
  local target_file=""
  if [[ -f "$HOME/.bashrc" || ! -f "$HOME/.bash_profile" ]]; then
    target_file="$HOME/.bashrc"
  else
    target_file="$HOME/.bash_profile"
  fi
  mkdir -p "$(dirname "$target_file")"
  touch "$target_file"

  python3 - "$target_file" "$qat_dir" <<'PY'
from pathlib import Path
import sys

target = Path(sys.argv[1])
qat_dir = sys.argv[2]
begin = "# >>> sima qat >>>"
end = "# <<< sima qat <<<"
block = f"""{begin}
activate-qat() {{
  source {qat_dir}/bin/activate
}}

deactivate-qat() {{
  deactivate 2>/dev/null || true
}}
{end}
"""
text = target.read_text(encoding="utf-8") if target.exists() else ""
start = text.find(begin)
finish = text.find(end)
if start != -1 and finish != -1 and finish > start:
    finish += len(end)
    text = text[:start].rstrip() + "\n\n" + block + text[finish:].lstrip("\n")
else:
    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n" + block
target.write_text(text, encoding="utf-8")
PY
}

if [[ ! -f "$SOURCE_JSON" ]]; then
  echo "Missing source manifest: $SOURCE_JSON" >&2
  exit 1
fi
if [[ ! -f "$WHEEL_MANIFEST" ]]; then
  echo "Missing wheel manifest: $WHEEL_MANIFEST" >&2
  exit 1
fi

PYTHON_VERSION_RAW="$(read_source_json_field python_version)"
if ! PYTHON_MM="$(normalize_python_version "$PYTHON_VERSION_RAW")"; then
  echo "Unsupported python_version in $SOURCE_JSON: '$PYTHON_VERSION_RAW'" >&2
  exit 1
fi
if ! PYTHON_CMD="$(resolve_python_cmd "$PYTHON_MM")"; then
  echo "Python $PYTHON_MM was not found. Install it or install pyenv and retry." >&2
  exit 1
fi

if [[ -d "/sdk-extensions" && -w "/sdk-extensions" ]]; then
  EXTENSIONS_DIR="/sdk-extensions"
elif [[ -d "/sdk-add-on" && -w "/sdk-add-on" ]]; then
  EXTENSIONS_DIR="/sdk-add-on"
else
  EXTENSIONS_DIR="$HOME/sdk-extensions"
  mkdir -p "$EXTENSIONS_DIR"
fi

QAT_DIR="$EXTENSIONS_DIR/qat"
echo "Creating QAT virtual environment at: $QAT_DIR (python: $PYTHON_CMD)"
reset_venv_dir "$QAT_DIR"
"$PYTHON_CMD" -m venv "$QAT_DIR"
"$QAT_DIR/bin/python" -m pip install --upgrade pip

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
  echo "No wheels listed in $WHEEL_MANIFEST" >&2
  exit 1
fi

pip_args=(--disable-pip-version-check --find-links "$BUNDLE_DIR")
if [[ -n "$EXTRA_INDEX_URL" ]]; then
  pip_args+=(--extra-index-url "$EXTRA_INDEX_URL")
fi
"$QAT_DIR/bin/python" -m pip install "${pip_args[@]}" "${wheels[@]}"
configure_shell_helpers "$QAT_DIR"
echo "QAT installation complete in $QAT_DIR."
