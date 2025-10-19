#!/usr/bin/env bash
set -euo pipefail

if command -v git >/dev/null 2>&1; then
  echo "Git is already installed at $(command -v git)"
  exit 0
fi

install_with_apt() {
  sudo apt update
  sudo apt install -y git
}

install_with_dnf() {
  sudo dnf install -y git
}

install_with_yum() {
  sudo yum install -y git
}

install_with_pacman() {
  sudo pacman -Sy --noconfirm git
}

install_with_apk() {
  sudo apk add git
}

install_with_zypper() {
  sudo zypper install -y git
}

install_with_brew() {
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required to install Git on macOS." >&2
    echo "Follow https://brew.sh/ to install Homebrew first." >&2
    exit 1
  fi
  brew update
  brew install git
}

OS_NAME=$(uname -s)

if [[ "${OS_NAME}" == "Darwin" ]]; then
  install_with_brew
  exit 0
fi

if [[ ! -f /etc/os-release ]]; then
  cat >&2 <<'MSG'
Unsupported Linux distribution: /etc/os-release is missing.
Install Git manually using your distribution's package manager.
MSG
  exit 1
fi

. /etc/os-release
case "${ID}" in
  ubuntu|debian|raspbian|linuxmint)
    install_with_apt
    ;;
  fedora)
    install_with_dnf
    ;;
  centos|rhel)
    if command -v dnf >/dev/null 2>&1; then
      install_with_dnf
    else
      install_with_yum
    fi
    ;;
  rocky|almalinux)
    install_with_dnf
    ;;
  arch|manjaro)
    install_with_pacman
    ;;
  alpine)
    install_with_apk
    ;;
  opensuse*|sles)
    install_with_zypper
    ;;
  *)
    cat >&2 <<EOF
Unsupported distribution ID: ${ID}
Install Git manually using your package manager.
EOF
    exit 1
    ;;
esac
