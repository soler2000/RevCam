#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/install.sh [options]

Create a Python virtual environment for RevCam and install the project.

Options:
  --python PATH           Python interpreter to use (default: python3)
  --venv PATH             Virtual environment directory (default: .venv)
  --pi                    Optimise installation for Raspberry Pi OS. Creates the
                          virtual environment with --system-site-packages and
                          installs dependencies through PiWheels when possible.
  --no-pi                 Disable Raspberry Pi auto-detection and keep the
                          virtual environment isolated.
  --dev                   Install the development dependencies defined in
                          pyproject.toml.
  --recreate              Delete any existing virtual environment before
                          creating a new one.
  -h, --help              Show this help message and exit.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV_DIR="$PROJECT_ROOT/.venv"
USE_PI=false
FORCE_NO_PI=false
AUTO_ENABLED_PI=false
WITH_DEV=false
RECREATE=false

VENV_FLAGS=()
PIP_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 1; }
            PYTHON="$2"
            shift 2
            ;;
        --venv)
            [[ $# -ge 2 ]] || { echo "--venv requires a value" >&2; exit 1; }
            case "$2" in
                /*) VENV_DIR="$2" ;;
                *) VENV_DIR="$PROJECT_ROOT/$2" ;;
            esac
            shift 2
            ;;
        --pi)
            USE_PI=true
            shift
            ;;
        --no-pi)
            FORCE_NO_PI=true
            shift
            ;;
        --dev)
            WITH_DEV=true
            shift
            ;;
        --recreate)
            RECREATE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
fi

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "RevCam requires Python 3.11 or newer." >&2
    exit 1
fi

if [[ "$USE_PI" == false && "$FORCE_NO_PI" == false ]]; then
    set +e
    "$PYTHON" - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("picamera2")
sys.exit(99 if spec is not None else 0)
PY
    status=$?
    set -e
    if [[ $status -eq 99 ]]; then
        echo "Detected system-wide Picamera2 installation; enabling Raspberry Pi integration (use --no-pi to disable)."
        USE_PI=true
        AUTO_ENABLED_PI=true
    elif [[ $status -ne 0 ]]; then
        echo "Warning: unable to determine Picamera2 availability (detector exit $status)" >&2
    fi
fi

if [[ "$USE_PI" == true ]]; then
    VENV_FLAGS+=(--system-site-packages)
    PIP_FLAGS+=(--prefer-binary --extra-index-url "https://www.piwheels.org/simple")
fi

if [[ "$RECREATE" == true && -d "$VENV_DIR" ]]; then
    echo "Removing existing virtual environment at $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment in $VENV_DIR"
    "$PYTHON" -m venv "${VENV_FLAGS[@]}" "$VENV_DIR"
else
    echo "Using existing virtual environment in $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

if [[ "$USE_PI" == true ]]; then
    if [[ "$AUTO_ENABLED_PI" == true && "$FORCE_NO_PI" == false ]]; then
        echo "Using Raspberry Pi friendly installation options (auto-detected system Picamera2 packages)."
    else
        echo "Enabling Raspberry Pi friendly installation options"
    fi
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip

INSTALL_TARGET="."
if [[ "$WITH_DEV" == true ]]; then
    INSTALL_TARGET=".[dev]"
fi

pushd "$PROJECT_ROOT" >/dev/null
trap 'popd >/dev/null' EXIT

set -x
"$VENV_DIR/bin/python" -m pip install "${PIP_FLAGS[@]}" -e "$INSTALL_TARGET"
set +x

# Ensure critical runtime dependencies that are imported at module load time are
# present in the environment even when the editable install is reused.
if ! "$VENV_DIR/bin/python" - <<'PY'
import importlib
import sys

try:
    importlib.import_module("numpy")
except ModuleNotFoundError:
    sys.exit(1)
else:
    sys.exit(0)
PY
then
    echo "Installing missing runtime dependency: numpy"
    set -x
    "$VENV_DIR/bin/python" -m pip install "${PIP_FLAGS[@]}" "numpy>=1.24"
    set +x
fi

if ! "$VENV_DIR/bin/python" - <<'PY'
import importlib
import sys

try:
    importlib.import_module("simplejpeg")
except ModuleNotFoundError:
    sys.exit(1)
else:
    sys.exit(0)
PY
then
    echo "Installing missing runtime dependency: simplejpeg"
    set -x
    "$VENV_DIR/bin/python" -m pip install "${PIP_FLAGS[@]}" "simplejpeg>=1.6"
    set +x
fi

if ! "$VENV_DIR/bin/python" - <<'PY'
import importlib
import sys

try:
    importlib.import_module("av")
except ModuleNotFoundError:
    sys.exit(1)
else:
    sys.exit(0)
PY
then
    echo "Installing missing runtime dependency: av"
    set -x
    "$VENV_DIR/bin/python" -m pip install "${PIP_FLAGS[@]}" "av>=10.0"
    set +x
fi

if ! "$VENV_DIR/bin/python" - <<'PY'
import importlib
import sys

try:
    importlib.import_module("aiortc")
except ModuleNotFoundError:
    sys.exit(1)
else:
    sys.exit(0)
PY
then
    echo "Installing missing runtime dependency: aiortc"
    set -x
    "$VENV_DIR/bin/python" -m pip install "${PIP_FLAGS[@]}" "aiortc>=1.7"
    set +x
fi

if ! "$VENV_DIR/bin/python" - <<'PY'
import importlib
import sys

try:
    importlib.import_module("board")
    importlib.import_module("neopixel")
except ModuleNotFoundError:
    sys.exit(1)
else:
    sys.exit(0)
PY
then
    echo "Installing LED ring driver dependencies: adafruit-blinka adafruit-circuitpython-neopixel"
    set -x
    "$VENV_DIR/bin/python" -m pip install "${PIP_FLAGS[@]}" --upgrade \
        adafruit-blinka \
        adafruit-circuitpython-neopixel
    set +x
fi

trap - EXIT
popd >/dev/null

cat <<SUMMARY

Installation complete.
Activate the virtual environment with:
  source "$VENV_DIR/bin/activate"

Launch the server (with sudo for NeoPixel access) via:
  ./scripts/run_with_sudo.sh

SUMMARY
