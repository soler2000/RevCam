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
    libjpeg-dev
    zlib1g-dev
    pkg-config
    libavdevice-dev
    libavfilter-dev
    libavformat-dev
    libavcodec-dev
    libavutil-dev
    libswresample-dev
    libswscale-dev
    libsrtp2-dev
    libopus-dev
    libvpx-dev
    libffi-dev
    libssl-dev
)

has_install_candidate() {
    local package="$1"
    local candidate

    candidate=$(apt-cache policy "$package" | awk '/Candidate:/ {print $2; exit}') || return 1

    [[ -n "$candidate" && "$candidate" != "(none)" ]]
}

atlas_package="libatlas-base-dev"
alt_atlas_packages=(libopenblas-dev liblapack-dev)

if [[ "$RUN_UPDATE" == true ]]; then
    echo "Updating package lists..."
    $SUDO apt update
fi

if has_install_candidate "$atlas_package"; then
    packages+=("$atlas_package")
else
    echo "Package $atlas_package is not available; installing ${alt_atlas_packages[*]} instead."
    packages+=("${alt_atlas_packages[@]}")
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

echo "Prerequisite installation complete."
