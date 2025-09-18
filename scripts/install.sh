#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/install.sh [OPTIONS]

Options:
  --pi            Install Raspberry Pi system packages and configure the
                  virtual environment to use them.
  --dev           Install development dependencies (pytest, etc.).
  --venv PATH     Location of the virtual environment directory (default: .venv).
  --python BIN    Python interpreter to use (default: python3).
  -h, --help      Show this help message and exit.
USAGE
}

PI_MODE=0
INSTALL_DEV=0
VENV_DIR=".venv"
PYTHON_BIN="python3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pi)
            PI_MODE=1
            shift
            ;;
        --dev)
            INSTALL_DEV=1
            shift
            ;;
        --venv)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Error: --venv requires a path" >&2
                exit 1
            fi
            VENV_DIR="$1"
            shift
            ;;
        --python)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Error: --python requires a binary name or path" >&2
                exit 1
            fi
            PYTHON_BIN="$1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            usage
            exit 1
            ;;
    esac
done

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

if [[ $PI_MODE -eq 1 ]]; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "Installing Raspberry Pi system dependencies (python3-picamera2, python3-prctl)..."
        sudo apt-get update
        sudo apt-get install -y python3-picamera2 python3-prctl
    else
        echo "Warning: apt-get not available; skipping Raspberry Pi system dependency installation" >&2
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR"
    VENV_FLAGS=()
    if [[ $PI_MODE -eq 1 ]]; then
        VENV_FLAGS+=("--system-site-packages")
    fi
    "$PYTHON_BIN" -m venv "${VENV_FLAGS[@]}" "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

pip_args=(install)
if [[ $PI_MODE -eq 1 ]]; then
    pip_args+=("--prefer-binary" "--extra-index-url" "https://www.piwheels.org/simple")
fi

TARGET="."
if [[ $INSTALL_DEV -eq 1 ]]; then
    TARGET=".[dev]"
fi

pip_args+=("-e" "$TARGET")

python -m pip "${pip_args[@]}"

echo
if [[ $PI_MODE -eq 1 ]]; then
    echo "RevCam is ready for Raspberry Pi with system camera packages available."
else
    echo "RevCam development environment is ready."
fi

deactivate
