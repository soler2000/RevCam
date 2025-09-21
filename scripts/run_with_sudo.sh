#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/run_with_sudo.sh [uvicorn-args]

Run the RevCam FastAPI application under sudo so the NeoPixel driver can access
/dev/mem. Additional arguments are forwarded to uvicorn. When no application
module is provided the script defaults to
  rev_cam.app:create_app --factory --host 0.0.0.0 --port 9000
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_VENV="$PROJECT_ROOT/.venv"
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV}"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $EUID -ne 0 ]]; then
    exec sudo --preserve-env=REVCAM_* VENV_DIR="$VENV_DIR" "$0" "$@"
fi

if [[ -d "$VENV_DIR" && -x "$VENV_DIR/bin/uvicorn" ]]; then
    UVICORN_BIN="$VENV_DIR/bin/uvicorn"
else
    UVICORN_BIN="$(command -v uvicorn || true)"
fi

if [[ -z "$UVICORN_BIN" ]]; then
    echo "Unable to locate uvicorn executable." >&2
    echo "Activate the virtual environment or set VENV_DIR to the desired path." >&2
    exit 1
fi

if [[ -d "$VENV_DIR/bin" ]]; then
    export PATH="$VENV_DIR/bin:$PATH"
fi

DEFAULT_ARGS=("rev_cam.app:create_app" "--factory" "--host" "0.0.0.0" "--port" "9000")
ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=("${DEFAULT_ARGS[@]}")
elif [[ "${ARGS[0]}" == -* ]]; then
    ARGS=("${DEFAULT_ARGS[@]}" "${ARGS[@]}")
fi

exec "$UVICORN_BIN" "${ARGS[@]}"
