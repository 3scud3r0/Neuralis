#!/usr/bin/env bash
# Isolated legacy emulator environment; never modifies the Studio toolchain.
set -euo pipefail
MARIO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MARIO_VENV="${MARIO_VENV:-$MARIO_ROOT/.venv}"
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x "$MARIO_VENV/bin/python" ]]; then
    uv venv --python 3.11 "$MARIO_VENV"
  fi
  "$MARIO_VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Mario requires Python 3.11"'
  if [[ "$(uname -s)" == Darwin ]]; then
    uv pip install --python "$MARIO_VENV/bin/python" 'torch==2.7.1'
  else
    uv pip install --python "$MARIO_VENV/bin/python" 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
  fi
  MARIO_REQUIREMENTS="$(mktemp)"
  trap 'rm -f "$MARIO_REQUIREMENTS"' EXIT
  sed '/^torch==/d' "$MARIO_ROOT/requirements.lock" > "$MARIO_REQUIREMENTS"
  uv pip install --python "$MARIO_VENV/bin/python" -r "$MARIO_REQUIREMENTS"
else
  if ! command -v python3.11 >/dev/null 2>&1; then
    printf '%s\n' 'Install Python 3.11 or uv, then rerun bootstrap.sh.' >&2
    exit 1
  fi
  if [[ ! -x "$MARIO_VENV/bin/python" ]]; then
    python3.11 -m venv "$MARIO_VENV"
  fi
  "$MARIO_VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Mario requires Python 3.11"'
  "$MARIO_VENV/bin/python" -m pip install --upgrade pip
  if [[ "$(uname -s)" == Darwin ]]; then
    "$MARIO_VENV/bin/python" -m pip install 'torch==2.7.1'
  else
    "$MARIO_VENV/bin/python" -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
  fi
  MARIO_REQUIREMENTS="$(mktemp)"
  trap 'rm -f "$MARIO_REQUIREMENTS"' EXIT
  sed '/^torch==/d' "$MARIO_ROOT/requirements.lock" > "$MARIO_REQUIREMENTS"
  "$MARIO_VENV/bin/python" -m pip install -r "$MARIO_REQUIREMENTS"
fi
cd "$MARIO_ROOT"
"$MARIO_VENV/bin/python" doctor.py
printf '\nActivate the environment with: source %q/bin/activate\n' "$MARIO_VENV"
