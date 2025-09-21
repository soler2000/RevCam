#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/install_prereqs.sh [options]

Install Raspberry Pi OS packages required for RevCam.

Options:
  --no-update   Skip running "apt update" before installing packages.
  -h, --help    Show this help message and exit.
USAGE
}

RUN_UPDATE=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-update)
            RUN_UPDATE=false
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

if ! command -v apt >/dev/null 2>&1; then
    echo "This script requires apt (Raspberry Pi OS / Debian-based systems)." >&2
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
    SUDO="sudo"
else
    SUDO=""
fi

packages=(
    git
    python3
    python3-venv
    python3-pip
    python3-picamera2
    python3-prctl
    libffi-dev
    libssl-dev
    libopus-dev
    libvpx-dev
    libavcodec-dev
    libavdevice-dev
    libavfilter-dev
    libavformat-dev
    libavutil-dev
    libswscale-dev
    libsrtp2-dev
    libatlas-base-dev
    libjpeg-dev
    zlib1g-dev
    pkg-config
)

if [[ "$RUN_UPDATE" == true ]]; then
    echo "Updating package lists..."
    $SUDO apt update
fi

echo "Installing prerequisites: ${packages[*]}"
$SUDO apt install -y "${packages[@]}"

echo "Ensuring SimpleJPEG is available (python3-simplejpeg)"
set +e
$SUDO apt install -y python3-simplejpeg
status=$?
set -e
if [[ $status -ne 0 ]]; then
    cat <<'NOTICE'
Warning: python3-simplejpeg is not available via APT on this system.
After creating the RevCam virtual environment, install SimpleJPEG with:
  source .venv/bin/activate
  python -m pip install --prefer-binary simplejpeg
If you are using a Raspberry Pi, add "--extra-index-url https://www.piwheels.org/simple"
to the pip command to reuse pre-built wheels.
NOTICE
else
    echo "python3-simplejpeg installed successfully."
fi

cat <<'INFO'

The additional FFmpeg development headers (libav*), codecs, and SRTP support
installed above allow pip to compile PyAV and aiortc when pre-built wheels are
unavailable. If either dependency fails to build later, rerun this script to
ensure the packages are up to date before retrying the Python installation.
INFO

echo "Prerequisite installation complete."
