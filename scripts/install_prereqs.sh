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
    libatlas-base-dev
    libavcodec-dev
    libavdevice-dev
    libavformat-dev
    libavutil-dev
    libswresample-dev
    libswscale-dev
    libopus-dev
    libvpx-dev
    libsrtp2-dev
    pkg-config
)

if [[ "$RUN_UPDATE" == true ]]; then
    echo "Updating package lists..."
    $SUDO apt update
fi

echo "Installing prerequisites: ${packages[*]}"
$SUDO apt install -y "${packages[@]}"

echo "Prerequisite installation complete."
